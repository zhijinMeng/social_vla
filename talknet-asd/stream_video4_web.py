#!/usr/bin/env python3
import argparse
import atexit
import json
import socket
import threading
import time
from http import server
from socketserver import ThreadingMixIn

import cv2


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>video4 live</title>
  <style>
    body {{
      margin: 0;
      background: #111;
      color: #eee;
      font-family: sans-serif;
    }}
    .wrap {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px;
    }}
    .meta {{
      margin-bottom: 16px;
      color: #bbb;
      line-height: 1.6;
    }}
    img {{
      width: 100%;
      height: auto;
      display: block;
      background: #000;
      border-radius: 12px;
    }}
    code {{
      background: #1d1d1d;
      padding: 2px 6px;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>/dev/video4 实时画面</h1>
    <div class="meta">
      打开地址：<code>{root_url}</code><br>
      MJPEG 地址：<code>{stream_url}</code>
    </div>
    <div class="meta">
      帧序号：<code id="frame-id">-</code><br>
      服务端采集时间：<code id="capture-ts">-</code><br>
      网页端估计延迟：<code id="latency-ms">-</code><br>
      网页本地时间：<code id="client-ts">-</code>
    </div>
    <img src="/stream.mjpg" alt="live stream">
  </div>
  <script>
    async function refreshStats() {{
      try {{
        const resp = await fetch('/stats.json', {{ cache: 'no-store' }});
        if (!resp.ok) return;
        const stats = await resp.json();
        const now = Date.now();
        const latency = (typeof stats.capture_ts_ms === 'number') ? (now - stats.capture_ts_ms) : null;
        document.getElementById('frame-id').textContent = stats.frame_id ?? '-';
        document.getElementById('capture-ts').textContent =
          (typeof stats.capture_ts_ms === 'number')
            ? new Date(stats.capture_ts_ms).toLocaleString() + ' (' + stats.capture_ts_ms + ' ms)'
            : '-';
        document.getElementById('latency-ms').textContent =
          (latency !== null) ? (latency + ' ms') : '-';
        document.getElementById('client-ts').textContent =
          new Date(now).toLocaleString() + ' (' + now + ' ms)';
      }} catch (e) {{
      }}
    }}
    refreshStats();
    setInterval(refreshStats, 500);
  </script>
</body>
</html>
"""


class CameraStream:
    def __init__(self, camera_path: str, width: int, height: int, fps: int, jpeg_quality: int):
        self.camera_path = camera_path
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.cap = None
        self.lock = threading.Lock()
        self.frame_jpeg = None
        self.capture_ts_ms = None
        self.frame_id = 0
        self.running = False
        self.thread = None

    def open(self) -> None:
        cap = cv2.VideoCapture(self.camera_path, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_path)
        if not cap.isOpened():
            raise RuntimeError(f"failed to open camera: {self.camera_path}")

        fourcc = cv2.VideoWriter_fourcc(*"YUYV")
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap = cap

    def start(self) -> None:
        self.open()
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _reader(self) -> None:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            capture_ts_ms = int(time.time() * 1000)
            self.frame_id += 1
            overlay = frame.copy()
            cv2.putText(
                overlay,
                f"frame={self.frame_id} ts={capture_ts_ms}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            ok, encoded = cv2.imencode(".jpg", overlay, encode_param)
            if not ok:
                continue
            with self.lock:
                self.frame_jpeg = encoded.tobytes()
                self.capture_ts_ms = capture_ts_ms

    def get_stats(self):
        with self.lock:
            return {
                "frame_id": self.frame_id,
                "capture_ts_ms": self.capture_ts_ms,
            }

    def get_frame(self):
        with self.lock:
            return self.frame_jpeg


class ThreadingHTTPServer(ThreadingMixIn, server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def build_handler(cam: CameraStream, host: str, port: int):
    class Handler(server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                root_url = f"http://{host}:{port}/"
                stream_url = f"http://{host}:{port}/stream.mjpg"
                body = HTML_PAGE.format(root_url=root_url, stream_url=stream_url).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/healthz":
                body = b"ok\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/stats.json":
                body = json.dumps(cam.get_stats()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        frame = cam.get_frame()
                        if frame is None:
                            time.sleep(0.05)
                            continue
                        self.wfile.write(b"--frame\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        time.sleep(max(1.0 / max(cam.fps, 1), 0.01))
                except (BrokenPipeError, ConnectionResetError):
                    return

            self.send_error(404)

        def log_message(self, fmt, *args):
            return

    return Handler


def infer_host_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def main():
    parser = argparse.ArgumentParser(description="Expose /dev/video4 as a simple MJPEG web page")
    parser.add_argument("--camera-path", default="/dev/video4")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    args = parser.parse_args()

    cam = CameraStream(
        camera_path=args.camera_path,
        width=args.width,
        height=args.height,
        fps=args.fps,
        jpeg_quality=args.jpeg_quality,
    )
    cam.start()
    atexit.register(cam.stop)

    display_host = infer_host_ip() if args.host == "0.0.0.0" else args.host
    print(f"[INFO] camera      : {args.camera_path}")
    print(f"[INFO] listen      : http://{args.host}:{args.port}/")
    print(f"[INFO] browser url : http://{display_host}:{args.port}/")
    print(f"[INFO] mjpeg url   : http://{display_host}:{args.port}/stream.mjpg")

    httpd = ThreadingHTTPServer((args.host, args.port), build_handler(cam, display_host, args.port))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        cam.stop()


if __name__ == "__main__":
    main()
