#!/usr/bin/env python3
"""Generate an original stylized-kid character archive for kid.rib.

Classic cartoon-child proportions — big cranium, soft wide face, large
low-set eyes, small nose and ears — built entirely from render.rs
primitives: a procedurally sculpted head mesh (lat-long grid deformed by
smooth radial fields), a scalp hair groom (cubic b-spline curves), and
quadrics for eyes/nose/ears/mouth.

Writes tests/fixtures/kid_character.rib (a self-contained archive with
materials) which kid.rib places and lights.
"""

import math
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "kid_character.rib"

# ---------------------------------------------------------------------------
# Head sculpt: radius field over (theta from +y pole, phi around y).
# Face looks toward +z.

def smooth(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def gauss(d2, s):
    return math.exp(-d2 / (2 * s * s))


def head_radius(theta, phi):
    """Radius at a direction; also used to place hair roots and features."""
    # Unit direction
    y = math.cos(theta)
    r_xy = math.sin(theta)
    x = r_xy * math.sin(phi)
    z = r_xy * math.cos(phi)  # +z = face front

    r = 1.0
    # Wide face, slightly shallow front-to-back.
    r *= math.sqrt(1.0 / ((x / 1.14) ** 2 + (y / 1.02) ** 2 + (z / 0.94) ** 2 + 1e-9)) \
        if False else 1.0
    # (analytic ellipsoid via direction scaling instead:)
    inv = math.sqrt((x / 1.18) ** 2 + (y / 1.06) ** 2 + (z / 0.94) ** 2)
    r = 1.0 / max(inv, 1e-6)

    # Jaw taper: lower half narrows softly toward a small chin.
    if y < 0.0:
        r *= 1.0 - 0.16 * smooth(-y / 0.95) * (0.4 + 0.6 * max(z, 0.0))
    # Big soft cheeks: bulges low on the face front.
    for sx in (-1.0, 1.0):
        d2 = (x - sx * 0.52) ** 2 + (y + 0.32) ** 2 + (z - 0.72) ** 2
        r *= 1.0 + 0.13 * gauss(d2, 0.45)
    # Chin bump.
    d2 = x ** 2 + (y + 0.88) ** 2 + (z - 0.42) ** 2
    r *= 1.0 + 0.05 * gauss(d2, 0.30)
    # Eye sockets: gentle dents where the big eyes nest.
    for sx in (-1.0, 1.0):
        d2 = (x - sx * 0.47) ** 2 + (y - 0.01) ** 2 + (z - 0.88) ** 2
        r *= 1.0 - 0.10 * gauss(d2, 0.30)
    # Brow/forehead stays full (big cranium).
    d2 = x ** 2 + (y - 0.55) ** 2 + (z - 0.75) ** 2
    r *= 1.0 + 0.045 * gauss(d2, 0.55)
    return r


def surface_point(theta, phi):
    r = head_radius(theta, phi)
    y = math.cos(theta) * r
    r_xy = math.sin(theta) * r
    x = r_xy * math.sin(phi) * 1.0
    z = r_xy * math.cos(phi)
    return (x, y, z)


def build_head(nu=96, nv=72):
    """Lat-long grid over the sculpt; returns (points, counts, indices,
    normals) with shared pole vertices avoided by tiny polar caps."""
    pts = []
    for j in range(nv + 1):
        theta = math.pi * j / nv
        for i in range(nu):
            phi = 2 * math.pi * i / nu
            pts.append(surface_point(theta, phi))
    counts, indices = [], []
    for j in range(nv):
        for i in range(nu):
            a = j * nu + i
            b = j * nu + (i + 1) % nu
            c = (j + 1) * nu + (i + 1) % nu
            d = (j + 1) * nu + i
            counts.append(4)
            indices.extend((a, b, c, d))
    # Smooth normals: accumulate face normals.
    normals = [[0.0, 0.0, 0.0] for _ in pts]
    for f in range(len(counts)):
        ia, ib, ic, _id = indices[f * 4:f * 4 + 4]
        pa, pb, pc = pts[ia], pts[ib], pts[ic]
        ux, uy, uz = (pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2])
        vx, vy, vz = (pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2])
        n = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
        for idx in (ia, ib, ic, _id):
            normals[idx][0] += n[0]
            normals[idx][1] += n[1]
            normals[idx][2] += n[2]
    for n in normals:
        l = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2) or 1.0
        n[0] /= l
        n[1] /= l
        n[2] /= l
    return pts, counts, indices, normals


# ---------------------------------------------------------------------------
# Hair groom: short tousled strands over the scalp region.

def hair(strands=30000):
    """Combed hair: strands follow a coherent direction field (swept to
    the side and forward), lie along the scalp in an arc, vary in length
    (longer on top), and the hairline waves instead of cutting a bowl
    edge. Clump-ish coherence comes from a smooth spatial jitter field.
    """
    import random
    random.seed(11)
    nv, pieces = [], []
    n = 0
    tries = 0
    while n < strands and tries < strands * 8:
        tries += 1
        u = random.random()
        theta = math.acos(1 - u)  # denser near the crown
        phi = random.uniform(0, 2 * math.pi)
        y = math.cos(theta)
        x = math.sin(theta) * math.sin(phi)
        z = math.sin(theta) * math.cos(phi)
        # Wavy hairline: higher in front, lower behind, broken by a wave.
        hairline = (0.30 + 0.36 * max(z, 0.0)
                    + 0.035 * math.sin(phi * 6.0 + 1.3)
                    + 0.03 * math.sin(phi * 11.0 + 4.0)
                    + 0.02 * math.sin(phi * 17.0 + 0.7)
                    + random.uniform(-0.035, 0.035))
        if y < hairline:
            continue
        px, py, pz = surface_point(theta, phi)
        nl = math.sqrt(px * px + py * py + pz * pz) or 1.0
        nx, ny, nz = px / nl, py / nl, pz / nl

        # Comb field: swept toward +x and slightly forward, with a smooth
        # spatial swirl so tufts flow together, plus per-strand jitter.
        cx = (0.85 + 0.35 * math.sin(3.1 * pz + 1.7)
              + 0.22 * math.sin(7.3 * px + 2.9 * pz))
        cy = -0.15 + 0.15 * math.sin(5.1 * px - 3.7 * pz + 1.1)
        cz = (0.40 + 0.35 * math.cos(2.7 * px - 0.6)
              + 0.22 * math.cos(6.1 * pz - 4.3 * px))
        cx += random.uniform(-0.3, 0.3)
        cy += random.uniform(-0.12, 0.12)
        cz += random.uniform(-0.3, 0.3)
        # Project into the tangent plane (hair lies along the scalp).
        d = cx * nx + cy * ny + cz * nz
        tx, ty, tz = cx - d * nx, cy - d * ny, cz - d * nz
        tl = math.sqrt(tx * tx + ty * ty + tz * tz) or 1.0
        tx, ty, tz = tx / tl, ty / tl, tz / tl

        # Length: longest at the crown, shorter toward the hairline.
        crown = smooth((y - hairline) / max(1.0 - hairline, 1e-3))
        ln = (0.14 + 0.24 * crown) * random.uniform(0.65, 1.5)

        # Random lift: most strands hug the scalp, some arc well off it,
        # and a few flyaways poke through — the silhouette gets depth
        # instead of tracing the head's perfect curve.
        lk = random.uniform(0.5, 1.9)
        if random.random() < 0.05:
            lk = random.uniform(2.2, 3.2)
            ln *= random.uniform(1.1, 1.5)
        droop = random.uniform(0.12, 0.4)

        pts = []
        for k in range(4):
            t = k / 3.0
            # Arc: lift off the scalp early, settle back down.
            lift = ln * lk * (0.55 * t - 0.42 * t * t)
            along = ln * t
            jx = random.uniform(-0.012, 0.012) if 0 < k < 3 else 0.0
            jy = random.uniform(-0.012, 0.012) if 0 < k < 3 else 0.0
            jz = random.uniform(-0.012, 0.012) if 0 < k < 3 else 0.0
            fx = px + nx * lift + tx * along + jx
            fy = py + ny * lift + ty * along - droop * ln * t * t + jy
            fz = pz + nz * lift + tz * along + jz
            pts.append((fx, fy, fz))
        nv.append(4)
        pieces.extend(pts)
        n += 1
    return nv, pieces


# ---------------------------------------------------------------------------

def w(f, s):
    f.write(s + "\n")


def main():
    pts, counts, indices, normals = build_head()
    hair_nv, hair_pts = hair()

    with open(OUT, "w") as f:
        w(f, "# generated by scripts/gen_kid.py — original stylized kid")
        w(f, "AttributeBegin")

        # ---- skin ----
        w(f, 'Attribute "identifier" "name" ["kid_head"]')
        w(f, 'Bxdf "PxrSurface" "skin"')
        w(f, '    "diffuseGain" [0.55] "diffuseColor" [0.68 0.36 0.22]')
        w(f, '    "subsurfaceGain" [0.55]')
        w(f, '    "subsurfaceColor" [0.8 0.38 0.22]')
        w(f, '    "subsurfaceDmfp" [0.3 0.15 0.08]')
        w(f, '    "specularIor" [1.4] "specularRoughness" [0.38]')

        # head
        w(f, "PointsPolygons [" + " ".join(str(c) for c in counts) + "]")
        w(f, "  [" + " ".join(str(i) for i in indices) + "]")
        w(f, '  "P" [' + " ".join(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" for p in pts) + "]")
        w(f, '  "N" [' + " ".join(f"{n[0]:.4f} {n[1]:.4f} {n[2]:.4f}" for n in normals) + "]")

        # tiny nose
        nx, ny, nz = surface_point(math.radians(97), 0.0)
        w(f, "TransformBegin")
        w(f, f"Translate {nx:.3f} {ny + 0.02:.3f} {nz + 0.035:.3f}")
        w(f, "Scale 1.0 0.85 0.7")
        w(f, "Sphere 0.075 -0.075 0.075 360")
        w(f, "TransformEnd")

        # small ears
        for sx in (-1, 1):
            ex, ey, ez = surface_point(math.radians(92), sx * math.radians(90))
            w(f, "TransformBegin")
            w(f, f"Translate {ex * 1.02:.3f} {ey:.3f} {ez:.3f}")
            w(f, "Scale 0.45 1.0 0.75")
            w(f, "Sphere 0.1 -0.1 0.1 360")
            w(f, "TransformEnd")

        # ---- eyes: big, dark, wide apart, low on the face ----
        w(f, 'Bxdf "PxrSurface" "eye"')
        w(f, '    "diffuseGain" [0.16] "diffuseColor" [0.055 0.032 0.02]')
        w(f, '    "specularIor" [1.7] "specularRoughness" [0.06]')
        for sx in (-1, 1):
            # nest into the sockets
            w(f, "TransformBegin")
            w(f, f"Translate {sx * 0.47:.3f} 0.01 0.82")
            w(f, "Sphere 0.22 -0.22 0.22 360")
            w(f, "TransformEnd")

        # ---- mouth: small soft smile (partial torus arc) ----
        w(f, 'Bxdf "PxrSurface" "mouth"')
        w(f, '    "diffuseGain" [0.85] "diffuseColor" [0.34 0.12 0.1]')
        w(f, '    "specularIor" [1.3] "specularRoughness" [0.35]')
        w(f, "TransformBegin")
        w(f, "Translate 0 -0.44 0.88")
        w(f, "Rotate 12 1 0 0")
        # bottom arc of a ring facing the camera = smile
        w(f, "Rotate 244 0 0 1")
        w(f, "Torus 0.19 0.018 0 360 52")
        w(f, "TransformEnd")

        # ---- hair ----
        w(f, 'Bxdf "PxrMarschnerHair" "hair"')
        w(f, '    "color" [0.45 0.26 0.11] "roughness" [0.24] "azimuthalRoughness" [0.35]')
        w(f, 'Basis "b-spline" 1 "b-spline" 1')
        w(f, 'Curves "cubic" [' + " ".join(str(v) for v in hair_nv) + '] "nonperiodic"')
        w(f, '  "P" [' + " ".join(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" for p in hair_pts) + "]")
        w(f, '  "width" [0.013 0.0035]')

        w(f, "AttributeEnd")

        # ---- body: rounded shoulders in a warm sweater ----
        w(f, "AttributeBegin")
        w(f, 'Attribute "identifier" "name" ["kid_body"]')
        w(f, 'Bxdf "PxrSurface" "sweater"')
        w(f, '    "diffuseGain" [0.85] "diffuseColor" [0.55 0.14 0.1]')
        w(f, '    "diffuseRoughness" [0.6] "specularIor" [1]')
        w(f, '    "fuzzGain" [1.1] "fuzzColor" [0.85 0.45 0.35]')
        w(f, "TransformBegin")
        w(f, "Translate 0 -1.82 0")
        w(f, "Scale 1.35 0.8 0.9")
        w(f, "Sphere 1.0 -1.0 1.0 360")
        w(f, "TransformEnd")
        # neck
        w(f, 'Bxdf "PxrSurface" "neck_skin"')
        w(f, '    "diffuseGain" [0.55] "diffuseColor" [0.68 0.36 0.22]')
        w(f, '    "subsurfaceGain" [0.55] "subsurfaceColor" [0.8 0.38 0.22]')
        w(f, '    "subsurfaceDmfp" [0.3 0.15 0.08]')
        w(f, '    "specularIor" [1.4] "specularRoughness" [0.38]')
        w(f, "TransformBegin")
        w(f, "Translate 0 -1.05 0")
        w(f, "Rotate -90 1 0 0")
        w(f, "Cylinder 0.34 -0.3 0.3 360")
        w(f, "TransformEnd")
        w(f, "AttributeEnd")

    size_mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT} ({len(pts)} head verts, {len(hair_nv)} strands, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
