__all__ = [
    "StreamingLAMEngine",
    "StreamingTalkNetEngine",
    "SimpleByteTracker",
    "YoloPersonDetector",
]


def __getattr__(name: str):
    if name == "StreamingLAMEngine":
        from social_vla.perception.lam_stream import StreamingLAMEngine

        return StreamingLAMEngine
    if name == "StreamingTalkNetEngine":
        from social_vla.perception.talknet_stream import StreamingTalkNetEngine

        return StreamingTalkNetEngine
    if name == "SimpleByteTracker":
        from social_vla.perception.yolo_tracker import SimpleByteTracker

        return SimpleByteTracker
    if name == "YoloPersonDetector":
        from social_vla.perception.yolo_tracker import YoloPersonDetector

        return YoloPersonDetector
    raise AttributeError(name)
