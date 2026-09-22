#!/usr/bin/env python3
"""Static dev server for the Kon Summon toy.

Plain `python -m http.server` lets the browser cache ES modules, so edits to
src/*.js silently do not take effect on reload. This sends no-store for
everything, which is what you want while iterating.

    python serve.py [port]        # default $PORT, else 8123
"""
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class NoCacheHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        '.js': 'text/javascript',
        '.mjs': 'text/javascript',
        '.glb': 'model/gltf-binary',
        '.webp': 'image/webp',
        '.wasm': 'application/wasm',
    }

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def log_message(self, fmt, *args):
        if '200' not in (args[1] if len(args) > 1 else ''):
            super().log_message(fmt, *args)


if __name__ == '__main__':
    # argv wins, then the PORT the preview tool assigns, then the default
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get('PORT', 8123))
    print(f'Kon Summon dev server: http://localhost:{port}')
    print(f'  mock camera (no webcam needed): http://localhost:{port}/?mock=1')
    ThreadingHTTPServer(('127.0.0.1', port), NoCacheHandler).serve_forever()
