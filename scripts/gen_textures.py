#!/usr/bin/env python3
"""Generate the Phase 6 demo texture set (pure stdlib, deterministic):

- wood_1001/1002/1011/1012.png: one continuous 512x512 parquet floor
  sliced into a 2x2 UDIM tile set (256px per tile)
- label.png: a bright generated poster for the crate

Run from the repo root: python3 scripts/gen_textures.py
Outputs land in tests/fixtures/.
"""

import math
import struct
import zlib
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def write_png(path, width, height, rgb):
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    raw = b""
    for y in range(height):
        raw += b"\x00" + bytes(rgb[y * width * 3:(y + 1) * width * 3])
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9))
           + chunk(b"IEND", b""))
    path.write_bytes(png)
    print(f"wrote {path} ({width}x{height})")


def hash01(ix, iy):
    h = (ix * 374761393 + iy * 668265263) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFF) / 65535.0


def smooth_noise(x, y):
    ix, iy = int(math.floor(x)), int(math.floor(y))
    fx, fy = x - ix, y - iy
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    a = hash01(ix, iy)
    b = hash01(ix + 1, iy)
    c = hash01(ix, iy + 1)
    d = hash01(ix + 1, iy + 1)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def fbm(x, y, octaves=4):
    total, amp, freq = 0.0, 1.0, 1.0
    norm = 0.0
    for _ in range(octaves):
        total += amp * smooth_noise(x * freq, y * freq)
        norm += amp
        amp *= 0.5
        freq *= 2.1
    return total / norm


def parquet(size=512):
    """Herringbone-ish planks with wood grain."""
    px = bytearray(size * size * 3)
    plank_w = size // 12  # planks across
    plank_l = size // 3
    for y in range(size):
        for x in range(size):
            row = y // plank_w
            offset = (row % 2) * plank_l // 2
            col = (x + offset) // plank_l
            # Per-plank tone variation.
            tone = 0.72 + 0.28 * hash01(col * 7 + 1, row * 13 + 5)
            # Grain: stretched noise along the plank.
            g = fbm((x + offset) * 0.9 / 6.0 + col * 31.7, y * 0.9 / 1.4 + row * 17.3, 4)
            grain = 0.82 + 0.18 * math.sin(g * 14.0 + (x + offset) * 0.05)
            # Dark seams between planks.
            by = y % plank_w
            bx = (x + offset) % plank_l
            seam = 1.0
            if by < 2 or by >= plank_w - 2 or bx < 2 or bx >= plank_l - 2:
                seam = 0.45
            v = tone * grain * seam
            r = min(1.0, 0.42 * v + 0.075)
            g2 = min(1.0, 0.26 * v + 0.04)
            b = min(1.0, 0.135 * v + 0.02)
            i = (y * size + x) * 3
            # sRGB-encode (textures are ingested as sRGB).
            for k, c in enumerate((r, g2, b)):
                enc = 1.055 * (c ** (1 / 2.4)) - 0.055 if c > 0.0031308 else 12.92 * c
                px[i + k] = max(0, min(255, int(enc * 255 + 0.5)))
    return px


def label(size=256):
    """Bright circular 'shipping label' poster."""
    px = bytearray(size * size * 3)
    cx = cy = size / 2
    for y in range(size):
        for x in range(size):
            dx, dy = (x - cx) / size, (y - cy) / size
            r = math.hypot(dx, dy)
            ang = math.atan2(dy, dx)
            # Cream background, teal ring, red center, spokes.
            col = (0.93, 0.88, 0.78)
            if r < 0.42:
                col = (0.10, 0.45, 0.50) if int((ang / math.pi * 8) + 16) % 2 == 0 else (0.16, 0.55, 0.60)
            if r < 0.30:
                col = (0.92, 0.90, 0.84)
            if r < 0.20:
                col = (0.75, 0.15, 0.12)
            if r < 0.08:
                col = (0.95, 0.85, 0.55)
            # Border stripes.
            edge = min(x, y, size - 1 - x, size - 1 - y) / size
            if edge < 0.04 and int((x + y) / 12) % 2 == 0:
                col = (0.15, 0.15, 0.2)
            i = (y * size + x) * 3
            for k, c in enumerate(col):
                enc = 1.055 * (c ** (1 / 2.4)) - 0.055 if c > 0.0031308 else 12.92 * c
                px[i + k] = max(0, min(255, int(enc * 255 + 0.5)))
    return px


def main():
    size = 1024
    full = parquet(size)
    half = size // 2
    # Slice into 2x2 UDIM tiles. UDIM tile (0,0)=1001 covers st in
    # [0,1)x[0,1) which is the BOTTOM-left in st space = bottom half of
    # the image (rows half..size, since images store top-down).
    names = {(0, 0): "wood_1001", (1, 0): "wood_1002", (0, 1): "wood_1011", (1, 1): "wood_1012"}
    for (sx, ty), name in names.items():
        tile = bytearray(half * half * 3)
        for y in range(half):
            src_y = (1 - ty) * half + y
            src = (src_y * size + sx * half) * 3
            tile[y * half * 3:(y + 1) * half * 3] = full[src:src + half * 3]
        write_png(OUT / f"{name}.png", half, half, tile)

    write_png(OUT / "label.png", 256, 256, label(256))


if __name__ == "__main__":
    main()
