# InferenceResult.msg - 推理结果消息定义
#
# 定义远程 GPU 推理管线 (Orin → 4090 via gRPC) 的输出格式。
# 包含模型标识、检测输出和通用 JSON 结果字段。
#
# 作者: The Robotic Science Team
# 日期: 2025-11-20
# 版本: 1.0

uint64 time_stamp
string model_name
float32[] detections    # 检测结果（格式取决于模型）
string result_json      # 通用 JSON 结果，灵活输出
