// Shared MSL: structs, math helpers, and primitive intersectors used
// by both the Whitted kernel (kernel.metal) and the path-tracing
// kernel (kernel_pt.metal). The host concatenates this file ahead of
// each kernel source before runtime compilation.

#include <metal_stdlib>
using namespace metal;

constant float T_EPS      = 1e-6f;   // CPU EPSILON hit threshold
constant float SHADOW_EPS = 1e-4f;   // shadow origin offset + occluder margin
constant float REFL_EPS   = 1e-4f;   // reflection origin offset

struct Object {
    // 0 sphere, 1 cylinder, 2 cone, 3 torus, 4 disk, 5 paraboloid,
    // 6 hyperboloid, 7 triangle
    uint  kind;
    float params[9];          // per-kind packing, see scene_buffers.rs
    float inv[16];            // row-major world->local
    float fwd[16];            // row-major local->world
};

// ---- math helpers ----

// Vec3::normalize parity: only divide when length > 1e-6.
inline float3 normalize_cpu(float3 v) {
    float len = length(v);
    return (len > 1e-6f) ? v / len : v;
}

// Affine transform as three float4 rows (bottom row [0,0,0,1] asserted at
// flatten time).
struct Affine { float4 r0, r1, r2; };

inline Affine load_affine(device const float* m) {
    Affine a;
    a.r0 = float4(m[0], m[1], m[2],  m[3]);
    a.r1 = float4(m[4], m[5], m[6],  m[7]);
    a.r2 = float4(m[8], m[9], m[10], m[11]);
    return a;
}
inline float3 xf_point(Affine a, float3 p) {
    float4 h = float4(p, 1.0f);
    return float3(dot(a.r0, h), dot(a.r1, h), dot(a.r2, h));
}
inline float3 xf_vec(Affine a, float3 v) {
    return float3(dot(a.r0.xyz, v), dot(a.r1.xyz, v), dot(a.r2.xyz, v));
}
// Matrix4::transform_normal: multiply by the transpose of the (inverse)
// matrix's upper 3x3.
inline float3 xf_normal(Affine inv, float3 n) {
    return float3(inv.r0.x * n.x + inv.r1.x * n.y + inv.r2.x * n.z,
                  inv.r0.y * n.x + inv.r1.y * n.y + inv.r2.y * n.z,
                  inv.r0.z * n.x + inv.r1.z * n.y + inv.r2.z * n.z);
}

// phi in [0, 360) from local x/y; accept when phi <= thetamax (CPU rule).
inline bool theta_ok(float x, float y, float thetamax) {
    float phi = atan2(y, x) * (180.0f / M_PI_F);
    if (phi < 0.0f) phi += 360.0f;
    return phi <= thetamax;
}

// ---- intersection ----

struct Hit { bool valid; float t; float3 p; float3 n; bool front; };

inline Hit miss_hit() {
    Hit h;
    h.valid = false; h.t = INFINITY; h.p = float3(0.0f); h.n = float3(0.0f);
    h.front = true;
    return h;
}

// World-space hit from a local hit point + local normal; t is the world
// distance from the ray origin (the CPU t convention).
inline Hit finish_hit(float3 world_origin, float3 local_p, float3 local_n,
                      Affine fwd, Affine inv) {
    Hit h;
    h.valid = true;
    float3 wp = xf_point(fwd, local_p);
    h.t = length(wp - world_origin);
    h.p = wp;
    h.n = normalize_cpu(xf_normal(inv, local_n));
    return h;
}

inline Hit isect_sphere(float3 wo, float3 lo, float3 ld, Affine fwd, Affine inv,
                        float radius, float zmin, float zmax, float thetamax) {
    float a = dot(ld, ld);                       // ~1 after normalize, kept for parity
    float b = 2.0f * dot(lo, ld);
    float c = dot(lo, lo) - radius * radius;
    float disc = b * b - 4.0f * a * c;
    if (disc < 0.0f) return miss_hit();
    float sq = sqrt(disc);
    float t1 = (-b - sq) / (2.0f * a);
    float t2 = (-b + sq) / (2.0f * a);
    // Near root, far-root fallback for rays starting inside (glass).
    float t1_sel = (t1 > T_EPS) ? t1 : t2;
    if (t1_sel <= T_EPS) return miss_hit();
    float3 p = lo + ld * t1_sel;
    if (p.z < zmin || p.z > zmax) return miss_hit();
    if (thetamax < 360.0f && !theta_ok(p.x, p.y, thetamax)) return miss_hit();
    return finish_hit(wo, p, normalize_cpu(p), fwd, inv);
}

inline Hit isect_cylinder(float3 wo, float3 lo, float3 ld, Affine fwd, Affine inv,
                          float radius, float zmin, float zmax, float thetamax) {
    float a = ld.x * ld.x + ld.y * ld.y;
    float b = 2.0f * (lo.x * ld.x + lo.y * ld.y);
    float c = lo.x * lo.x + lo.y * lo.y - radius * radius;
    if (fabs(a) < 1e-6f) return miss_hit();      // ray parallel to the axis
    float disc = b * b - 4.0f * a * c;
    if (disc < 0.0f) return miss_hit();
    float sq = sqrt(disc);
    float t1 = (-b - sq) / (2.0f * a);
    float t2 = (-b + sq) / (2.0f * a);
    float t = (t1 > T_EPS) ? t1 : t2;
    if (t <= T_EPS) return miss_hit();
    float z = lo.z + ld.z * t;
    if (z < zmin || z > zmax) {
        // CPU retry: only when the failing t was the near root
        // (deliberate float == to mirror `t == t1`).
        if (t == t1 && t2 > T_EPS) {
            t = t2;
            z = lo.z + ld.z * t;
            if (z < zmin || z > zmax) return miss_hit();
        } else {
            return miss_hit();
        }
    }
    float3 p = lo + ld * t;
    if (thetamax < 360.0f && !theta_ok(p.x, p.y, thetamax)) return miss_hit();
    return finish_hit(wo, p, normalize_cpu(float3(p.x, p.y, 0.0f)), fwd, inv);
}

// Exact quadric-gradient normal, as Cone::create_intersection (apex r guard
// matches the MLX port).
inline Hit cone_hit(float3 wo, float3 p, Affine fwd, Affine inv,
                    float radius, float height) {
    float r = max(sqrt(p.x * p.x + p.y * p.y), 1e-20f);
    float3 n = normalize_cpu(float3(p.x / r, p.y / r, -(radius / height)));
    return finish_hit(wo, p, n, fwd, inv);
}

inline Hit isect_cone(float3 wo, float3 lo, float3 ld, Affine fwd, Affine inv,
                      float height, float radius, float thetamax) {
    float k = (radius / height) * (radius / height);
    float a = ld.x * ld.x + ld.y * ld.y - k * ld.z * ld.z;
    float b = 2.0f * (lo.x * ld.x + lo.y * ld.y - k * lo.z * ld.z);
    float c = lo.x * lo.x + lo.y * lo.y - k * lo.z * lo.z;

    if (fabs(a) < 1e-6f) {
        // Degenerate linear path: z-clipped but NO theta check (CPU quirk).
        if (fabs(b) < 1e-6f) return miss_hit();
        float t = -c / b;
        if (t <= T_EPS) return miss_hit();
        float3 p = lo + ld * t;
        if (p.z < 0.0f || p.z > height) return miss_hit();
        return cone_hit(wo, p, fwd, inv, radius, height);
    }

    float disc = b * b - 4.0f * a * c;
    if (disc < 0.0f) return miss_hit();
    float sq = sqrt(disc);
    float roots[2] = { (-b - sq) / (2.0f * a), (-b + sq) / (2.0f * a) };
    for (int i = 0; i < 2; i++) {                // z-clip AND theta fall through to t2
        float t = roots[i];
        if (t <= T_EPS) continue;
        float3 p = lo + ld * t;
        if (p.z < 0.0f || p.z > height) continue;
        if (thetamax < 360.0f && !theta_ok(p.x, p.y, thetamax)) continue;
        return cone_hit(wo, p, fwd, inv, radius, height);
    }
    return miss_hit();
}

// ---- P0 primitives: torus, disk, paraboloid, hyperboloid, triangle ----

// Newton-polished quartic roots (ascending); returns count, fills roots[4].
// f32 Ferrari is too unstable — seed from coarse scan + Newton instead.
inline int solve_quartic(float a4, float a3, float a2, float a1, float a0,
                         float tmax, thread float* roots) {
    // Coarse march over [T_EPS, tmax] looking for sign changes, then bisect
    // + Newton. Robust in f32 and plenty fast for a handful of tori.
    const int STEPS = 128;
    int count = 0;
    float dt = tmax / (float)STEPS;
    float t_prev = T_EPS;
    float f_prev = (((a4 * t_prev + a3) * t_prev + a2) * t_prev + a1) * t_prev + a0;
    for (int i = 1; i <= STEPS && count < 4; i++) {
        float t = dt * (float)i;
        float f = (((a4 * t + a3) * t + a2) * t + a1) * t + a0;
        if ((f_prev < 0.0f) != (f < 0.0f)) {
            // Bisection refine.
            float lo = t_prev, hi = t, flo = f_prev;
            for (int j = 0; j < 24; j++) {
                float mid = 0.5f * (lo + hi);
                float fm = (((a4 * mid + a3) * mid + a2) * mid + a1) * mid + a0;
                if ((flo < 0.0f) != (fm < 0.0f)) { hi = mid; } else { lo = mid; flo = fm; }
            }
            roots[count++] = 0.5f * (lo + hi);
        }
        t_prev = t;
        f_prev = f;
    }
    return count;
}

inline Hit isect_torus(float3 wo, float3 lo, float3 ld, Affine fwd, Affine inv,
                       float R, float r, float phimin, float phimax, float thetamax) {
    // Bound the search with the torus bounding sphere (radius R+r), and
    // re-origin the ray at the window start: smaller coefficients condition
    // the f32 quartic far better and the root march resolves finer.
    float bound = R + r;
    float mid = dot(-lo, ld);                          // t of closest approach
    float closest2 = dot(lo + ld * mid, lo + ld * mid);
    if (closest2 > bound * bound) return miss_hit();
    float half_span = sqrt(bound * bound - closest2);
    float t_enter = max(mid - half_span, 0.0f);
    float t_exit = mid + half_span;
    if (t_exit <= T_EPS) return miss_hit();

    float3 so = lo + ld * t_enter;                     // shifted origin
    float g = dot(ld, ld);
    float h = 2.0f * dot(so, ld);
    float i = dot(so, so) + R * R - r * r;
    float j = ld.x * ld.x + ld.y * ld.y;
    float k = 2.0f * (so.x * ld.x + so.y * ld.y);
    float l = so.x * so.x + so.y * so.y;
    float fr = 4.0f * R * R;

    float roots[4];
    int n = solve_quartic(g * g, 2.0f * g * h,
                          h * h + 2.0f * g * i - fr * j,
                          2.0f * h * i - fr * k,
                          i * i - fr * l, t_exit - t_enter, roots);
    for (int q = 0; q < n; q++) {
        float t = roots[q] + t_enter;
        if (t <= T_EPS) continue;
        float3 p = lo + ld * t;
        float r_xy = sqrt(p.x * p.x + p.y * p.y);
        if (phimax - phimin < 360.0f) {
            float phi = atan2(p.z, r_xy - R) * (180.0f / M_PI_F);
            while (phi < phimin) phi += 360.0f;
            while (phi >= phimin + 360.0f) phi -= 360.0f;
            if (phi > phimax) continue;
        }
        if (thetamax < 360.0f && !theta_ok(p.x, p.y, thetamax)) continue;
        float scale = 1.0f - R / max(r_xy, 1e-12f);
        float3 nrm = normalize_cpu(float3(p.x * scale, p.y * scale, p.z));
        return finish_hit(wo, p, nrm, fwd, inv);
    }
    return miss_hit();
}

inline Hit isect_disk(float3 wo, float3 lo, float3 ld, Affine fwd, Affine inv,
                      float height, float radius, float thetamax) {
    if (fabs(ld.z) < T_EPS) return miss_hit();
    float t = (height - lo.z) / ld.z;
    if (t <= T_EPS) return miss_hit();
    float3 p = lo + ld * t;
    if (p.x * p.x + p.y * p.y > radius * radius) return miss_hit();
    if (thetamax < 360.0f && !theta_ok(p.x, p.y, thetamax)) return miss_hit();
    return finish_hit(wo, p, float3(0.0f, 0.0f, 1.0f), fwd, inv);
}

inline Hit isect_paraboloid(float3 wo, float3 lo, float3 ld, Affine fwd, Affine inv,
                            float rmax, float zmin, float zmax, float thetamax) {
    if (fabs(zmax) < T_EPS) return miss_hit();
    float k = rmax * rmax / zmax;
    float a = ld.x * ld.x + ld.y * ld.y;
    float b = 2.0f * (lo.x * ld.x + lo.y * ld.y) - k * ld.z;
    float c = lo.x * lo.x + lo.y * lo.y - k * lo.z;
    float t1, t2;
    if (fabs(a) < 1e-6f) {
        if (fabs(b) < 1e-6f) return miss_hit();
        t1 = -c / b; t2 = -1.0f;
    } else {
        float disc = b * b - 4.0f * a * c;
        if (disc < 0.0f) return miss_hit();
        float s = sqrt(disc);
        t1 = (-b - s) / (2.0f * a);
        t2 = (-b + s) / (2.0f * a);
    }
    float cand[2] = { t1, t2 };
    for (int q = 0; q < 2; q++) {
        float t = cand[q];
        if (t <= T_EPS) continue;
        float3 p = lo + ld * t;
        if (p.z < zmin || p.z > zmax) continue;
        if (thetamax < 360.0f && !theta_ok(p.x, p.y, thetamax)) continue;
        float3 nrm = normalize_cpu(float3(2.0f * p.x, 2.0f * p.y, -k));
        return finish_hit(wo, p, nrm, fwd, inv);
    }
    return miss_hit();
}

inline Hit isect_hyperboloid(float3 wo, float3 lo, float3 ld, Affine fwd, Affine inv,
                             float3 p1, float3 p2, float thetamax) {
    float zlo = min(p1.z, p2.z);
    float zhi = max(p1.z, p2.z);
    if (fabs(p2.z - p1.z) < T_EPS) {
        // Flat annulus between the two radii.
        if (fabs(ld.z) < T_EPS) return miss_hit();
        float t = (p1.z - lo.z) / ld.z;
        if (t <= T_EPS) return miss_hit();
        float3 p = lo + ld * t;
        float r2 = p.x * p.x + p.y * p.y;
        float ra = p1.x * p1.x + p1.y * p1.y;
        float rb = p2.x * p2.x + p2.y * p2.y;
        if (r2 < min(ra, rb) || r2 > max(ra, rb)) return miss_hit();
        if (thetamax < 360.0f && !theta_ok(p.x, p.y, thetamax)) return miss_hit();
        return finish_hit(wo, p, float3(0.0f, 0.0f, 1.0f), fwd, inv);
    }
    float u = 1.0f / (p2.z - p1.z);
    float sx = (p2.x - p1.x) * u;
    float sy = (p2.y - p1.y) * u;
    float cx = p1.x - p1.z * sx;
    float cy = p1.y - p1.z * sy;
    float A = sx * sx + sy * sy;
    float B = 2.0f * (cx * sx + cy * sy);
    float C = cx * cx + cy * cy;

    float a = ld.x * ld.x + ld.y * ld.y - A * ld.z * ld.z;
    float b = 2.0f * (lo.x * ld.x + lo.y * ld.y - A * lo.z * ld.z) - B * ld.z;
    float c = lo.x * lo.x + lo.y * lo.y - A * lo.z * lo.z - B * lo.z - C;
    float t1, t2;
    if (fabs(a) < 1e-6f) {
        if (fabs(b) < 1e-6f) return miss_hit();
        t1 = -c / b; t2 = -1.0f;
    } else {
        float disc = b * b - 4.0f * a * c;
        if (disc < 0.0f) return miss_hit();
        float s = sqrt(disc);
        t1 = (-b - s) / (2.0f * a);
        t2 = (-b + s) / (2.0f * a);
    }
    float cand[2] = { t1, t2 };
    for (int q = 0; q < 2; q++) {
        float t = cand[q];
        if (t <= T_EPS) continue;
        float3 p = lo + ld * t;
        if (p.z < zlo || p.z > zhi) continue;
        if (thetamax < 360.0f && !theta_ok(p.x, p.y, thetamax)) continue;
        float3 nrm = normalize_cpu(float3(2.0f * p.x, 2.0f * p.y, -(2.0f * A * p.z + B)));
        return finish_hit(wo, p, nrm, fwd, inv);
    }
    return miss_hit();
}

// Möller–Trumbore in object space; double-sided (normal faces the ray).
inline Hit isect_triangle(float3 wo, float3 lo, float3 ld, Affine fwd, Affine inv,
                          float3 v0, float3 v1, float3 v2) {
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 pvec = cross(ld, e2);
    float det = dot(e1, pvec);
    if (fabs(det) < 1e-12f) return miss_hit();
    float inv_det = 1.0f / det;
    float3 tvec = lo - v0;
    float uu = dot(tvec, pvec) * inv_det;
    if (uu < 0.0f || uu > 1.0f) return miss_hit();
    float3 qvec = cross(tvec, e1);
    float vv = dot(ld, qvec) * inv_det;
    if (vv < 0.0f || uu + vv > 1.0f) return miss_hit();
    float t = dot(e2, qvec) * inv_det;
    if (t <= T_EPS) return miss_hit();
    float3 p = lo + ld * t;
    float3 nrm = normalize_cpu(cross(e1, e2));
    // Record the geometric side before the double-sided flip.
    bool front = dot(nrm, ld) < 0.0f;
    if (!front) nrm = -nrm;
    Hit h = finish_hit(wo, p, nrm, fwd, inv);
    h.front = front;
    return h;
}

inline Hit isect_object(device const Object& obj, float3 wo, float3 wd) {
    Affine inv = load_affine(obj.inv);
    Affine fwd = load_affine(obj.fwd);
    float3 lo = xf_point(inv, wo);
    float3 ld = normalize_cpu(xf_vec(inv, wd));  // CPU normalizes local direction
    device const float* p = obj.params;
    Hit h;
    switch (obj.kind) {
        case 0u: h = isect_sphere(wo, lo, ld, fwd, inv, p[0], p[1], p[2], p[3]); break;
        case 1u: h = isect_cylinder(wo, lo, ld, fwd, inv, p[0], p[1], p[2], p[3]); break;
        case 2u: h = isect_cone(wo, lo, ld, fwd, inv, p[0], p[1], p[2]); break;
        case 3u: h = isect_torus(wo, lo, ld, fwd, inv, p[0], p[1], p[2], p[3], p[4]); break;
        case 4u: h = isect_disk(wo, lo, ld, fwd, inv, p[0], p[1], p[2]); break;
        case 5u: h = isect_paraboloid(wo, lo, ld, fwd, inv, p[0], p[1], p[2], p[3]); break;
        case 6u: h = isect_hyperboloid(wo, lo, ld, fwd, inv,
                                       float3(p[0], p[1], p[2]),
                                       float3(p[3], p[4], p[5]), p[6]); break;
        default: return isect_triangle(wo, lo, ld, fwd, inv,
                                       float3(p[0], p[1], p[2]),
                                       float3(p[3], p[4], p[5]),
                                       float3(p[6], p[7], p[8]));
    }
    // Quadrics report unflipped (outward) normals: the geometric side falls
    // out of the dot with the ray direction.
    if (h.valid) h.front = dot(h.n, wd) < 0.0f;
    return h;
}

