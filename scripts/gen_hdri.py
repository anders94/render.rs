#!/usr/bin/env python3
"""Generate a small procedural sky HDRI (Radiance .hdr, flat RGBE) for the
shaderball demo: gradient sky, sun disk, warm horizon, dark ground.

Usage: gen_hdri.py OUT.hdr [--width 512]
"""

import argparse
import math
import struct


def sky(u, v):
    """Radiance for lat-long coords u in [0,1), v in [0,1] (v=0 zenith)."""
    theta = v * math.pi          # polar angle from +Y
    phi = u * 2.0 * math.pi
    y = math.cos(theta)

    # Sun direction: 35 degrees up, azimuth 40 degrees.
    sun_el = math.radians(35)
    sun_az = math.radians(40)
    sy = math.sin(sun_el)
    sx = math.cos(sun_el) * math.sin(sun_az)
    sz = -math.cos(sun_el) * math.cos(sun_az)

    st = math.sin(theta)
    dx = st * math.sin(phi)
    dy = y
    dz = -st * math.cos(phi)
    cos_sun = dx * sx + dy * sy + dz * sz

    if y < -0.02:
        # Ground bounce: dim warm grey.
        g = 0.18 + 0.1 * max(0.0, 1.0 + y)
        return (g * 1.0, g * 0.95, g * 0.9)

    # Sky gradient: zenith blue to pale horizon.
    t = max(0.0, min(1.0, 1.0 - y))
    r = 0.18 + 0.75 * t ** 2.2
    g = 0.32 + 0.62 * t ** 2.0
    b = 0.65 + 0.30 * t
    # Warm glow near horizon toward the sun.
    glow = max(0.0, cos_sun) ** 8 * max(0.0, 1.0 - abs(y) * 4.0)
    r += 1.6 * glow
    g += 0.9 * glow
    b += 0.3 * glow
    # Sun disk (~2 degrees) with soft edge.
    if cos_sun > math.cos(math.radians(1.8)):
        edge = (cos_sun - math.cos(math.radians(1.8))) / (
            1.0 - math.cos(math.radians(1.8))
        )
        s = 400.0 * min(1.0, edge * 3.0)
        r += s
        g += s * 0.92
        b += s * 0.78
    return (r, g, b)


def rgbe(r, g, b):
    m = max(r, g, b)
    if m < 1e-32:
        return b"\x00\x00\x00\x00"
    e = math.frexp(m)[1]
    scale = math.ldexp(1.0, -e) * 256.0
    return struct.pack(
        "BBBB",
        min(255, int(r * scale)),
        min(255, int(g * scale)),
        min(255, int(b * scale)),
        e + 128,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--width", type=int, default=512)
    args = ap.parse_args()
    w = args.width
    h = w // 2

    with open(args.out, "wb") as f:
        f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(f"-Y {h} +X {w}\n".encode())
        for yy in range(h):
            v = (yy + 0.5) / h
            for xx in range(w):
                u = (xx + 0.5) / w
                f.write(rgbe(*sky(u, v)))
    print(f"wrote {args.out} ({w}x{h})")


if __name__ == "__main__":
    main()
