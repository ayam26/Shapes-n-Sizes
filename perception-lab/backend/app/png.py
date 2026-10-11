"""Minimal PNG validation and metadata stripping (no Pillow dependency).

Only IHDR, PLTE, tRNS, IDAT and IEND survive, so descriptive text chunks
(tEXt/iTXt/zTXt, timestamps, EXIF) never reach the model or the hash."""
import struct
import zlib

SIG = b"\x89PNG\r\n\x1a\n"
KEEP = {b"IHDR", b"PLTE", b"tRNS", b"IDAT", b"IEND"}


class PNGError(ValueError):
    pass


def strip_and_validate(data: bytes, max_dim: int) -> tuple[bytes, int, int]:
    if not data.startswith(SIG):
        raise PNGError("not a PNG file")
    pos, out, width, height, seen_end = len(SIG), [SIG], None, None, False
    while pos < len(data):
        if pos + 8 > len(data):
            raise PNGError("truncated chunk header")
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        end = pos + 12 + length
        if end > len(data):
            raise PNGError("truncated chunk")
        body = data[pos + 8:pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length:end])
        if zlib.crc32(ctype + body) & 0xFFFFFFFF != crc:
            raise PNGError(f"bad CRC in {ctype!r}")
        if ctype == b"IHDR":
            width, height = struct.unpack(">II", body[:8])
        if ctype in KEEP:
            out.append(data[pos:end])
        pos = end
        if ctype == b"IEND":
            seen_end = True
            break
    if width is None or not seen_end:
        raise PNGError("missing IHDR or IEND")
    if not (0 < width <= max_dim and 0 < height <= max_dim):
        raise PNGError(f"image dimensions {width}x{height} outside 1..{max_dim}")
    return b"".join(out), width, height


def solid_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Valid RGB PNG of one colour (used for the unrecorded warm-up call and tests)."""
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    return (SIG + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
