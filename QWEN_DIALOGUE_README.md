# QwenOmni 实时对话系统

## 快速启动

```bash
cd /home/zhijinmeng/Research/social_vla_engage
./run_qwen.sh
```

或者手动运行：

```bash
cd /home/zhijinmeng/Research/social_vla_engage
source /home/zhijinmeng/Research/HF_Data/percy_data/.venv/bin/activate
export $(grep -v '^#' /home/zhijinmeng/Research/.env.local | grep -v '^$' | xargs)

python -W ignore social_vla/pipeline/live_demo.py \
  --camera 0 --device cuda \
  --mic 4 --vad-backend silero --audio-gain 2.0 \
  --score-threshold 0.45 \
  --qwen
```

## 系统架构

### 音频流处理

1. **麦克风输入** → Silero VAD 检测语音活动
2. **VAD Gating** → `QwenOmniAdapter.gate_vad()` 过滤回声和非监听状态
3. **音频缓冲** → 
   - Preroll: 保存最近的音频用于 VAD 触发前的内容
   - Turn audio: 用户回合期间的音频实时入队
4. **后台流式发送** → `_audio_stream_loop` 线程持续将音频发送到 API
5. **对话提交** → VAD 结束时提交，触发 LLM 响应

### 关键修复（2026-06-11）

#### 问题：音频丢失导致 "user audio too short"

**根本原因**：音频队列有两个消费者竞争：
- `_audio_stream_loop`（后台线程）：持续从队列取音频发送到 API
- `_flush_audio_queue`（提交时）：试图再次取队列中的音频

后台线程已经把音频都取走了，导致 `_flush_audio_queue` 返回 0 bytes。

**解决方案**：
1. 移除 `_flush_audio_queue` 中的队列消费逻辑
2. 使用计数器 `_turn_audio_bytes` 追踪本轮对话累积的字节数
3. `_enqueue_audio` 时增加计数器
4. `_flush_audio_queue` 只需等待队列排空，返回计数器值

**相关代码变更**：
- `qwen_omni_adapter.py:200`: 添加 `_turn_audio_bytes` 计数器
- `qwen_omni_adapter.py:449`: `begin_user_turn` 时重置计数器
- `qwen_omni_adapter.py:507`: `_enqueue_audio` 时累加计数器
- `qwen_omni_adapter.py:516`: `_flush_audio_queue` 返回计数器值而非消费队列

## VAD 状态机

```
listening=True, in_turn=False    ← 初始状态，等待用户说话
    ↓ (VAD 触发)
listening=False, in_turn=True    ← 用户回合，录制音频
    ↓ (VAD 结束)
listening=False, in_turn=False   ← 提交 API，等待机器人回复
    ↓ (机器人说完 + echo guard)
listening=True, in_turn=False    ← 循环回初始状态
```

### Gate VAD 逻辑

```python
if playing:           # 机器人在说话
    return False      # 禁止 VAD（防止回声）
elif in_turn:         # 用户回合中
    return raw        # 接受 VAD 信号（继续录音）
elif not listening:   # Echo guard 期间
    return False      # 禁止 VAD
else:
    return raw        # 正常传递 VAD
```

## 配置参数

### 麦克风设备

列出可用设备：
```bash
python social_vla/pipeline/live_demo.py --list-devices
```

选择设备：`--mic <设备编号>`（示例中使用 `--mic 4`）

### VAD 参数

- `--vad-backend silero`: 使用 Silero VAD（推荐）
- `--audio-gain 2.0`: 麦克风增益（提高灵敏度）

### 对话参数

- `--score-threshold 0.45`: Engagement 阈值，触发主动对话
- `--qwen`: 启用 QwenOmni 实时对话模型

## 调试

### 日志级别

所有关键日志使用 Python `logging`，可通过环境变量控制：
```bash
export LOG_LEVEL=DEBUG
./run_qwen.sh
```

### 常见问题

**Q: 机器人不回应我的话**
- 检查 VAD 是否触发：看状态行中 `vad=1` 是否出现
- 检查音频增益：尝试增加 `--audio-gain 3.0`
- 查看日志：`[QwenOmni] flushed X bytes` 应该 > 9600

**Q: 机器人一直重复同样的话**
- **原因**：对话历史累积过长，API 进入自动重复模式
- **解决**：按 `r` 键重置会话（清除对话历史）
- **预防**：建议每 20-30 轮对话后按 `r` 重置一次

**Q: "user audio too short" 错误**
- 这个问题已修复（见上文）
- 如果仍然出现，检查 `_turn_audio_bytes` 是否正确累加

**Q: 回声抑制不工作**
- 确认 `gate_vad` 在 `playing=True` 时返回 False
- 调整 `echo_guard_ms` 参数（默认 1500ms）

## 依赖

- Python 3.12+
- PyTorch + CUDA
- dashscope SDK (QwenOmni API)
- sounddevice (音频播放)
- pyaudio (麦克风输入)
- silero-vad (语音活动检测)
- YOLO (视觉检测)

## API 配置

需要在 `.env.local` 中配置：
```
DASHSCOPE_API_KEY=your_key_here
```
