"""vllm-ui entrypoint.

Without TLS: plain uvicorn on UI_HOST:UI_PORT — nothing else.

With TLS (UI_TLS_ACTIVE set by run-ui.sh): the public port must serve BOTH
protocols so that http://domain:8090 redirects instead of showing a browser
TLS error. A TLS ClientHello always starts with record-type byte 0x16, while
a plain HTTP request starts with an ASCII method — so a small TCP front on
UI_HOST:UI_PORT peeks the first byte and either
  * splices the connection byte-for-byte to uvicorn (https), which listens
    with the certificate on loopback UI_PORT+1 (never exposed), or
  * reads the request line and answers `301 Location: https://DOMAIN:PORT/…`.
The splice is a dumb bidirectional pump, so streaming (SSE chat, log follow)
passes through untouched.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn

UI_HOST = os.environ.get("UI_HOST", "0.0.0.0").strip() or "0.0.0.0"
UI_PORT = int(os.environ.get("UI_PORT", "8090").strip() or "8090")
UI_DOMAIN = os.environ.get("UI_DOMAIN", "").strip()
TLS_ACTIVE = bool(os.environ.get("UI_TLS_ACTIVE", "").strip())
INTERNAL_PORT = UI_PORT + 1   # loopback-only TLS listener behind the demux
CERT_BASE = os.path.join("ui", "data", "certs", "live", UI_DOMAIN)


async def _pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError, OSError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    try:
        first = await asyncio.wait_for(reader.read(1), timeout=10)
        if not first:
            writer.close()
            return
        if first == b"\x16":          # TLS handshake → splice to uvicorn
            up_r, up_w = await asyncio.open_connection("127.0.0.1", INTERNAL_PORT)
            up_w.write(first)
            await asyncio.gather(_pump(reader, up_w), _pump(up_r, writer))
            return
        # plain HTTP → one 301 and close
        line = first + await asyncio.wait_for(reader.readline(), timeout=10)
        try:
            path = line.split()[1].decode("latin-1")
        except (IndexError, UnicodeDecodeError):
            path = "/"
        if not path.startswith("/"):
            path = "/"
        writer.write((
            "HTTP/1.1 301 Moved Permanently\r\n"
            f"Location: https://{UI_DOMAIN}:{UI_PORT}{path}\r\n"
            "Content-Length: 0\r\nConnection: close\r\n\r\n").encode("latin-1"))
        await writer.drain()
        writer.close()
    except (asyncio.TimeoutError, ConnectionError, OSError):
        try:
            writer.close()
        except Exception:
            pass


async def main():
    if not TLS_ACTIVE:
        cfg = uvicorn.Config("app:app", host=UI_HOST, port=UI_PORT,
                             access_log=False)
        await uvicorn.Server(cfg).serve()
        return
    demux = await asyncio.start_server(_handle, UI_HOST, UI_PORT)
    cfg = uvicorn.Config("app:app", host="127.0.0.1", port=INTERNAL_PORT,
                         access_log=False,
                         ssl_certfile=os.path.join(CERT_BASE, "fullchain.pem"),
                         ssl_keyfile=os.path.join(CERT_BASE, "privkey.pem"))
    print(f"[serve] https on {UI_HOST}:{UI_PORT} "
          f"(http on the same port redirects)", flush=True)
    async with demux:
        await asyncio.gather(demux.serve_forever(), uvicorn.Server(cfg).serve())


if __name__ == "__main__":
    asyncio.run(main())
