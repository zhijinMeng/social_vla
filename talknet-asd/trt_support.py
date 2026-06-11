import math
from pathlib import Path

import cv2
import numpy as np
import python_speech_features
import tensorrt as trt
import torch

from model.faceDetector.s3fd.box_utils import decode, nms_


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def _torch_dtype_from_trt(dtype):
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT8: torch.int8,
        trt.DataType.BOOL: torch.bool,
    }
    if dtype not in mapping:
        raise RuntimeError(f"Unsupported TensorRT dtype: {dtype}")
    return mapping[dtype]


class TensorRTRunner:
    def __init__(self, engine_path: str):
        self.engine_path = str(engine_path)
        self.runtime = trt.Runtime(TRT_LOGGER)
        engine_bytes = Path(self.engine_path).read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create TensorRT execution context: {self.engine_path}")
        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_names = [n for n in self.tensor_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.output_names = [n for n in self.tensor_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

    def get_tensor_shape(self, name: str):
        return tuple(self.engine.get_tensor_shape(name))

    def run(self, feeds: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for name in self.input_names:
            if name not in feeds:
                raise RuntimeError(f"Missing TensorRT input: {name}")

        stream = torch.cuda.current_stream().cuda_stream
        bound = {}
        for name, tensor in feeds.items():
            if not tensor.is_cuda:
                tensor = tensor.cuda(non_blocking=True)
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            engine_shape = tuple(self.engine.get_tensor_shape(name))
            runtime_shape = tuple(tensor.shape)
            if -1 not in engine_shape and engine_shape != runtime_shape:
                raise RuntimeError(
                    f"TensorRT input shape mismatch for {name}: engine expects {engine_shape}, got {runtime_shape} "
                    f"(engine={self.engine_path})"
                )
            ok_shape = self.context.set_input_shape(name, runtime_shape)
            if ok_shape is False:
                raise RuntimeError(
                    f"Failed to set TensorRT input shape for {name}: {runtime_shape} "
                    f"(engine={self.engine_path})"
                )
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
            bound[name] = tensor

        try:
            all_specified = self.context.all_binding_shapes_specified
        except AttributeError:
            all_specified = True
        if not all_specified:
            raise RuntimeError(f"Not all TensorRT binding shapes were specified for engine {self.engine_path}")

        outputs = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(int(dim) < 0 for dim in shape):
                raise RuntimeError(
                    f"TensorRT output shape unresolved for {name}: {shape} (engine={self.engine_path})"
                )
            dtype = _torch_dtype_from_trt(self.engine.get_tensor_dtype(name))
            out = torch.empty(shape, device="cuda", dtype=dtype)
            self.context.set_tensor_address(name, int(out.data_ptr()))
            outputs[name] = out

        ok = self.context.execute_async_v3(stream)
        if not ok:
            raise RuntimeError(f"TensorRT execution failed: {self.engine_path}")
        torch.cuda.synchronize()
        return outputs


class S3FDTRTDetector:
    def __init__(self, engine_specs: dict[float, str], device="cuda"):
        self.device = device
        self.runners = {float(k): TensorRTRunner(v) for k, v in engine_specs.items()}
        self.input_name = "input"
        self.loc_name = "loc"
        self.conf_name = "conf"
        self.priors_name = "priors"
        self.img_mean = np.array([104.0, 117.0, 123.0])[:, np.newaxis, np.newaxis].astype("float32")
        self.variance = [0.1, 0.2]

    def detect_faces(self, image, conf_th=0.8, scales=[1]):
        w, h = image.shape[1], image.shape[0]
        bboxes = np.empty(shape=(0, 5), dtype=np.float32)

        for s in scales:
            s = float(s)
            if s not in self.runners:
                raise RuntimeError(
                    f"Missing S3FD TRT engine for scale {s}. Available scales: {sorted(self.runners.keys())}"
                )
            scaled_img = cv2.resize(image, dsize=(0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            scaled_img = np.swapaxes(scaled_img, 1, 2)
            scaled_img = np.swapaxes(scaled_img, 1, 0)
            scaled_img = scaled_img[[2, 1, 0], :, :]
            scaled_img = scaled_img.astype("float32")
            scaled_img -= self.img_mean
            scaled_img = scaled_img[[2, 1, 0], :, :]
            x = torch.from_numpy(scaled_img).unsqueeze(0).to(self.device)
            expected_shape = self.runners[s].get_tensor_shape(self.input_name)
            got_shape = tuple(int(v) for v in x.shape)
            if -1 not in expected_shape and tuple(int(v) for v in expected_shape) != got_shape:
                raise RuntimeError(
                    f"S3FD TRT engine/input mismatch for scale {s}: engine expects {expected_shape}, got {got_shape}. "
                    "This usually means camera resolution does not match the engine build resolution. "
                    "Set --camWidth/--camHeight to match the engine, or rebuild the engine for the actual frame size."
                )
            outputs = self.runners[s].run({self.input_name: x})
            loc = outputs[self.loc_name].detach().float().cpu()
            conf = outputs[self.conf_name].detach().float().cpu()
            priors = outputs[self.priors_name].detach().float().cpu()

            boxes = decode(loc.view(-1, 4), priors.view(-1, 4), self.variance)
            boxes = boxes.view(loc.size(0), -1, 4)[0]
            scores = conf[0, :, 1]
            mask = scores > float(conf_th)
            if not torch.any(mask):
                continue
            boxes = boxes[mask]
            scores = scores[mask]
            scale_tensor = torch.tensor([w, h, w, h], dtype=torch.float32)
            boxes = boxes * scale_tensor
            for box, score in zip(boxes.numpy(), scores.numpy()):
                bbox = (float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(score))
                bboxes = np.vstack((bboxes, bbox))

        if len(bboxes) == 0:
            return bboxes
        keep = nms_(bboxes, 0.1)
        return bboxes[keep]


def _pad_or_trim_2d(arr: np.ndarray, target_len: int):
    arr = np.asarray(arr)
    cur = arr.shape[0]
    if cur == target_len:
        return arr
    if cur > target_len:
        return arr[:target_len]
    pad = np.zeros((target_len - cur, arr.shape[1]), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def _pad_or_trim_3d(arr: np.ndarray, target_len: int):
    arr = np.asarray(arr)
    cur = arr.shape[0]
    if cur == target_len:
        return arr
    if cur > target_len:
        return arr[:target_len]
    if cur == 0:
        return np.zeros((target_len, *arr.shape[1:]), dtype=arr.dtype)
    pad = np.repeat(arr[-1:,:,:], target_len - cur, axis=0)
    return np.concatenate([arr, pad], axis=0)


class TalkNetTRTBackend:
    def __init__(self, engine_path: str, audio_sr: int):
        self.runner = TensorRTRunner(engine_path)
        self.audio_sr = int(audio_sr)
        self.audio_name = "audio"
        self.video_name = "video"
        self.output_name = "scores"
        audio_shape = self.runner.get_tensor_shape(self.audio_name)
        video_shape = self.runner.get_tensor_shape(self.video_name)
        self.expected_audio_frames = int(audio_shape[1])
        self.expected_video_frames = int(video_shape[1])

    def infer_scores(self, audio_seg, video_frames):
        if len(video_frames) < 4 or len(audio_seg) < 160:
            return None

        audio_feature = python_speech_features.mfcc(
            audio_seg,
            self.audio_sr,
            numcep=13,
            winlen=0.025,
            winstep=0.010,
        )
        length = min((audio_feature.shape[0] - audio_feature.shape[0] % 4) / 100.0, video_frames.shape[0] / 25.0)
        if length <= 0:
            return None
        audio_feature = audio_feature[: int(round(length * 100)), :]
        video_frames = video_frames[: int(round(length * 25)), :, :]
        valid_video_len = max(1, min(video_frames.shape[0], self.expected_video_frames))

        audio_feature = _pad_or_trim_2d(audio_feature, self.expected_audio_frames).astype(np.float32)
        video_frames = _pad_or_trim_3d(video_frames, self.expected_video_frames).astype(np.float32)

        input_a = torch.from_numpy(audio_feature).unsqueeze(0).cuda()
        input_v = torch.from_numpy(video_frames).unsqueeze(0).cuda()
        out = self.runner.run({self.audio_name: input_a, self.video_name: input_v})[self.output_name]
        scores = out.detach().float().cpu().numpy().reshape(-1)
        return scores[:valid_video_len]
