# SocialVLA-Engage

展会机器人 **「确认过眼神」** 项目 — Phase 1 Layer A 感知与互动触发原型。

目标：在多人场景下判断「是否在对机器人说话 / 是否有交流意图」，并为后续 Qwen2.5-VL（Layer B）提供结构化感知输入。

## 当前进度（2026-06）

### 已完成（W1）

| 模块 | 状态 | 说明 |
|------|------|------|
| YOLO + ByteTrack | ✅ | `yolov8n` 人体检测 + IoU 跟踪 |
| LAM 流式推理 | ✅ | 7 帧滑窗；缺权重时 mock fallback |
| TalkNet 流式推理 | ✅ | 1s AV 滑窗；权重 `pretrain_TalkSet.model` |
| Engagement Score | ✅ | 无声偏 LAM、有声偏 TalkNet 动态融合 |
| Session + Trigger | ✅ | robot / user / turn 三路触发状态机 |
| `mock_demo.py` | ✅ | 合成画面离线验证 |
| `live_demo.py` | ✅ | 摄像头真人检测 + 预览叠框 |
| Silero VAD | ✅ | 复用 PERCY `percy_dialogue` 同款逻辑 |
| 麦克风采集 | ✅ | `sounddevice`；硬件不支持 16kHz 时自动 44.1k→16k 重采样 |

### 未做 / 进行中（W2+）

| 模块 | 状态 | 说明 |
|------|------|------|
| LAM 官方权重 | ⏳ | 需 Ego4D 申请 → `weights/lam_gazelstm.pth` |
| ROS 接入 | ⏳ | head cam / `/audio/rode` / TTS topic |
| Layer B (VL-7B) | ⏳ | vLLM + faster-whisper |
| 数据采集 pipeline | ⏳ | rosbag + session metadata |
| Docker 交付 | ⏳ | PyTorch/CUDA 与皮任东对齐 |

## 架构（Layer A @ 10Hz）

```
Camera + Mic
    ↓
YoloPersonDetector + SimpleByteTracker
    ↓ per track
StreamingLAMEngine (7-frame)     StreamingTalkNetEngine (1s AV)
    ↓                                   ↓
EngagementScorer（无声→LAM 权重↑，有声→TalkNet 权重↑）
    ↓
TriggerRouter → SessionManager
```

## 环境

推荐复用已有 MERCI venv，或新建 Python 3.11+：

```bash
# 系统依赖（麦克风）
sudo apt install libportaudio2 portaudio19-dev

pip install -r requirements.txt
bash scripts/setup_third_party.sh   # TalkNet-ASD + ego4d-lam
bash scripts/download_weights.sh    # TalkNet 公开权重
```

| 依赖 | 版本参考 |
|------|----------|
| Python | 3.11+ |
| PyTorch | 2.12 + CUDA 13.x（本机 MERCI venv） |
| ultralytics | 8.4+ |
| silero-vad | 5.1+（与 PERCY 一致） |

权重文件不入库，本地放置：

- `weights/pretrain_TalkSet.model` — TalkNet（`download_weights.sh`）
- `weights/yolov8n.pt` — 首次运行 YOLO 自动下载，或放此路径
- `weights/lam_gazelstm.pth` — Ego4D LAM（需申请）

## 快速运行

### Mock（无摄像头）

```bash
python social_vla/pipeline/mock_demo.py --ticks 80 --vad-mock
python social_vla/pipeline/mock_demo.py --real-weights --device cuda --ticks 80 --vad-mock
```

### Live 真人 + 说话

```bash
# 列出麦克风
python social_vla/pipeline/live_demo.py --list-devices

# 内置麦（ROG ALC294 用 --mic 4）
python social_vla/pipeline/live_demo.py \
  --camera 0 --device cuda --perception-only \
  --mic 4 --vad-backend silero --audio-gain 2.0
```

- `q` 退出，`r` 重置 session
- `--perception-only`：只打分，显示 `WOULD_TRIGGER`，不改 state
- 去掉 `--perception-only` 可测完整触发流程

### 日志字段

```
#128 | state=idle | vad=1 | id=1 eng=0.71 lam=0.70m talk=0.35m | rms=1200
```

| 字段 | 含义 |
|------|------|
| `lam` | 目光 / look-at-me 概率；`m`=缓冲就绪 |
| `talk` | TalkNet 说话概率 |
| `vad` | Silero 是否在说话 |
| `rms` | 麦克风音量（0=未采集到音频） |

## 目录结构

```
social_vla/
  perception/   # YOLO、LAM、TalkNet、VAD、麦克风
  scoring/      # Engagement 融合
  session/      # 状态机 + 触发器
  pipeline/     # layer_a、mock_demo、live_demo
scripts/        # third_party 安装、权重下载
configs/        # 默认超参
tests/
third_party/    # git clone（不入库）
weights/        # 本地权重（不入库）
```

## 相关仓库

- PERCY 对话 / Silero VAD：`../percy_ws/src/percy_dialogue/`
- 架构设计：内部文档 SocialVLA-Engage 端侧算法架构（B5-v1）

## License

第三方模型遵循各自仓库协议（TalkNet、Ego4D）。本仓库代码 MIT。
