#!/usr/bin/env python3
"""Export ONNX and build TensorRT engines for S3FD and TalkNet.

Examples:
  python3 build_trt_engines.py --mode both --pretrainModel pretrain_TalkSet.model \
    --s3fdBaseWidth 1920 --s3fdBaseHeight 1080 --s3fdScales 0.5,1.0,1.5 \
    --talknetAudioFrames 100 --talknetVideoFrames 25 --fp16
"""

import argparse
from pathlib import Path

import tensorrt as trt
import torch
import torch.nn.functional as F

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


class TalkNetTRTExport(torch.nn.Module):
    def __init__(self, speaker):
        super().__init__()
        self.model = speaker.model
        self.fc = speaker.lossAV.FC

    def forward(self, audio, video):
        embed_a = self.model.forward_audio_frontend(audio)
        embed_v = self.model.forward_visual_frontend(video)
        embed_a, embed_v = self.model.forward_cross_attention(embed_a, embed_v)
        out = self.model.forward_audio_visual_backend(embed_a, embed_v)
        logits = self.fc(out.squeeze(1))
        return logits[:, 1]


class S3FDTRTExport(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        size = x.size()[2:]
        sources = []
        loc = []
        conf = []

        for k in range(16):
            x = self.net.vgg[k](x)
        s = self.net.L2Norm3_3(x)
        sources.append(s)

        for k in range(16, 23):
            x = self.net.vgg[k](x)
        s = self.net.L2Norm4_3(x)
        sources.append(s)

        for k in range(23, 30):
            x = self.net.vgg[k](x)
        s = self.net.L2Norm5_3(x)
        sources.append(s)

        for k in range(30, len(self.net.vgg)):
            x = self.net.vgg[k](x)
        sources.append(x)

        for k, v in enumerate(self.net.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        loc_x = self.net.loc[0](sources[0])
        conf_x = self.net.conf[0](sources[0])
        max_conf, _ = torch.max(conf_x[:, 0:3, :, :], dim=1, keepdim=True)
        conf_x = torch.cat((max_conf, conf_x[:, 3:, :, :]), dim=1)

        loc.append(loc_x.permute(0, 2, 3, 1).contiguous())
        conf.append(conf_x.permute(0, 2, 3, 1).contiguous())

        for i in range(1, len(sources)):
            feat = sources[i]
            conf.append(self.net.conf[i](feat).permute(0, 2, 3, 1).contiguous())
            loc.append(self.net.loc[i](feat).permute(0, 2, 3, 1).contiguous())

        feature_maps = []
        for item in loc:
            feature_maps.append([item.size(1), item.size(2)])

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        conf = self.net.softmax(conf.view(conf.size(0), -1, 2))

        from model.faceDetector.s3fd.box_utils import PriorBox

        priors = PriorBox(size, feature_maps).forward().to(x.device)
        return loc.view(loc.size(0), -1, 4), conf, priors


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def build_engine_from_onnx(onnx_path: Path, engine_path: Path, fp16: bool, workspace_gb: float):
    ensure_parent(engine_path)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [parser.get_error(i) for i in range(parser.num_errors)]
        raise RuntimeError(f"Failed to parse ONNX: {onnx_path}\n" + "\n".join(map(str, errors)))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1024 ** 3)))
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"Failed to build TensorRT engine from {onnx_path}")
    engine_path.write_bytes(bytes(serialized))
    print(f"[ok] engine: {engine_path}")


def export_talknet(args, output_dir: Path):
    from talkNet import talkNet

    speaker = talkNet()
    speaker.loadParameters(args.pretrainModel)
    speaker.eval()

    model = TalkNetTRTExport(speaker).eval().cuda()
    audio = torch.randn(1, args.talknetAudioFrames, 13, device="cuda", dtype=torch.float32)
    video = torch.randn(1, args.talknetVideoFrames, 112, 112, device="cuda", dtype=torch.float32)

    onnx_path = output_dir / f"talknet_a{args.talknetAudioFrames}_v{args.talknetVideoFrames}.onnx"
    engine_path = output_dir / f"talknet_a{args.talknetAudioFrames}_v{args.talknetVideoFrames}.engine"
    ensure_parent(onnx_path)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (audio, video),
            str(onnx_path),
            input_names=["audio", "video"],
            output_names=["scores"],
            opset_version=args.opset,
            do_constant_folding=True,
        )
    print(f"[ok] onnx:   {onnx_path}")
    build_engine_from_onnx(onnx_path, engine_path, fp16=args.fp16, workspace_gb=args.workspaceGB)


def export_s3fd(args, output_dir: Path):
    from model.faceDetector.s3fd.nets import S3FDNet

    weights_path = Path(__file__).resolve().parent / "model" / "faceDetector" / "s3fd" / "sfd_face.pth"
    if not weights_path.exists():
        raise RuntimeError(
            f"S3FD weights not found: {weights_path}. "
            "Run the original S3FD path once or place sfd_face.pth there before building TRT."
        )
    net = S3FDNet(device="cuda").cuda()
    state_dict = torch.load(str(weights_path), map_location="cuda")
    net.load_state_dict(state_dict)
    net.eval()
    model = S3FDTRTExport(net).eval().cuda()

    for scale in parse_scales(args.s3fdScales):
        height = max(32, int(round(args.s3fdBaseHeight * scale)))
        width = max(32, int(round(args.s3fdBaseWidth * scale)))
        dummy = torch.randn(1, 3, height, width, device="cuda", dtype=torch.float32)
        tag = f"{height}x{width}_s{scale}".replace(".", "p")
        onnx_path = output_dir / f"s3fd_{tag}.onnx"
        engine_path = output_dir / f"s3fd_{tag}.engine"
        ensure_parent(onnx_path)
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy,
                str(onnx_path),
                input_names=["input"],
                output_names=["loc", "conf", "priors"],
                opset_version=args.opset,
                do_constant_folding=True,
            )
        print(f"[ok] onnx:   {onnx_path}")
        build_engine_from_onnx(onnx_path, engine_path, fp16=args.fp16, workspace_gb=args.workspaceGB)


def parse_scales(raw: str):
    vals = []
    for item in str(raw).split(","):
        item = item.strip()
        if item:
            vals.append(float(item))
    if not vals:
        raise RuntimeError("No valid scales provided")
    return vals


def main():
    ap = argparse.ArgumentParser("Build TensorRT engines for TalkNet and S3FD")
    ap.add_argument("--mode", choices=["talknet", "s3fd", "both"], default="both")
    ap.add_argument("--pretrainModel", type=str, default="pretrain_TalkSet.model")
    ap.add_argument("--outDir", type=str, default="trt_engines")
    ap.add_argument("--fp16", action="store_true", help="Build FP16 engines when supported")
    ap.add_argument("--workspaceGB", type=float, default=4.0)
    ap.add_argument("--opset", type=int, default=17)

    ap.add_argument("--talknetAudioFrames", type=int, default=100, help="TalkNet audio MFCC frame count")
    ap.add_argument("--talknetVideoFrames", type=int, default=25, help="TalkNet visual frame count")

    ap.add_argument("--s3fdBaseWidth", type=int, default=1920)
    ap.add_argument("--s3fdBaseHeight", type=int, default=1080)
    ap.add_argument("--s3fdScales", type=str, default="0.5,1.0,1.5")

    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to export and build TensorRT engines")

    root = Path(__file__).resolve().parent
    output_dir = (root / args.outDir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("talknet", "both"):
        export_talknet(args, output_dir)
    if args.mode in ("s3fd", "both"):
        export_s3fd(args, output_dir)

    print("\nRecommended online args:")
    if args.mode in ("talknet", "both"):
        talknet_engine = output_dir / f"talknet_a{args.talknetAudioFrames}_v{args.talknetVideoFrames}.engine"
        print(f"  --asdBackend talknet_trt --talknetTrtEngine {talknet_engine}")
    if args.mode in ("s3fd", "both"):
        specs = []
        for scale in parse_scales(args.s3fdScales):
            height = max(32, int(round(args.s3fdBaseHeight * scale)))
            width = max(32, int(round(args.s3fdBaseWidth * scale)))
            tag = f"{height}x{width}_s{scale}".replace('.', 'p')
            specs.append(f"{scale}={output_dir / f's3fd_{tag}.engine'}")
        print(f"  --faceDetectorBackend s3fd_trt --s3fdTrtEngineSpecs {','.join(specs)}")


if __name__ == "__main__":
    main()
