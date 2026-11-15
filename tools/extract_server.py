"""Small local server used only for building the dataset.

It serves the project folder (so the browser can load the SCUT images and
MediaPipe files) and accepts POST /save with landmark batches, which it writes
to data/landmarks/. Run from the project root:

    python tools/extract_server.py
then open http://localhost:8765/tools/extract.html
"""
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "landmarks")
os.makedirs(OUT, exist_ok=True)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        if self.path != "/save":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        name = os.path.basename(body["file"])  # keep writes inside OUT
        with open(os.path.join(OUT, name), "w") as f:
            json.dump(body["rows"], f)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")


if __name__ == "__main__":
    print("serving on http://localhost:8765")
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
