// Path-tracing kernel (roadmap Phase 3): a Metal port of the CPU
// reference integrator in src/raytracer/pt/mod.rs — NEE + MIS (power
// heuristic), Russian roulette, firefly clamp, Lambert/GGX lobes — plus
// nested TLAS/BLAS traversal for instanced meshes. Compiled after
// isect_common.metal (see renderer.rs). Struct layouts mirror
// gpu_scene.rs byte-for-byte (scalar fields only).
//
// f32 throughout; parity with the f64 CPU integrator is statistical
// (same light transport, independent float error), verified by
// mean/RMSE tests rather than per-pixel comparison.

constant float PT_RAY_OFFSET = 1e-4f;
constant float PT_PI = 3.14159265358979f;
// Finite stand-in for infinity: this kernel compiles with fast math ON
// (unlike the whitted kernel), so IEEE INF semantics cannot be relied on.
constant float PT_BIG = 1e30f;

struct PtUniforms {
    uint  width;
    uint  height;
    uint  sample_start;
    uint  sample_count;
    uint  object_count;
    uint  instance_count;
    uint  light_count;
    uint  max_bounces;
    uint  rr_start;
    uint  y_offset;
    float firefly_clamp;
    float background[3];
    float eye[3];
    float forward[3];
    float right[3];
    float up[3];
    float half_width;
    float half_height;
    uint  dome_index;       // 0xFFFFFFFF = none
    uint  env_width;
    uint  env_height;
    float env_total;
};

struct PtMaterial {
    float diffuse_gain;
    float diffuse_color[3];
    float diffuse_sigma;
    float spec_f0[3];
    float spec_f90[3];
    float spec_alpha;
    float coat_gain;
    float coat_alpha;
    float fuzz_gain;
    float fuzz_color[3];
    float glass_gain;
    float glass_ior;
    float glass_alpha;
    float refr_color[3];
    float emission[3];
    float presence;
    float under_scale;
    float weights[5];       // d, s, c, f, g (normalized; all zero = dead)
    uint  area_light;
};

struct PtLight {
    uint  kind;          // 0 point, 1 distant, 2 rect, 3 sphere, 4 disk, 5 dome
    float a[3];          // position / direction / corner
    float e1[3];
    float e2[3];
    float normal[3];
    float area;
    float radiance[3];
    float pad[2];
};

struct BvhNodeG {
    float mn[3];
    uint  left_or_first;
    float mx[3];
    uint  count;         // 0 interior, >0 leaf
};

struct MeshInfoG {
    uint node_offset;
    uint index_offset;
    uint vertex_offset;
    uint has_normals;
};

struct InstanceG {
    float inv[16];
    float fwd[16];
    uint  mesh_id;
    uint  material_id;
    uint  pad[2];
};

// All scene pointers bundled so helpers have sane signatures.
struct PtScene {
    device const Object*    objects;
    device const uint*      object_materials;
    device const PtMaterial* materials;
    device const PtLight*   lights;
    device const BvhNodeG*  tlas;
    device const InstanceG* instances;
    device const BvhNodeG*  blas;
    device const uint*      tri_indices;
    device const float*     vertices;
    device const float*     normals;
    device const MeshInfoG* mesh_infos;
    device const float*     env_pixels;
    device const float*     env_marginal;
    device const float*     env_conditional;
    constant PtUniforms*    u;
};

// ---- PCG32, bit-matching the Rust sampler's u32 stream ----

struct Pcg32 { ulong state; ulong inc; };

inline uint pcg_next(thread Pcg32& r) {
    ulong old = r.state;
    r.state = old * 6364136223846793005UL + r.inc;
    uint xorshifted = (uint)(((old >> 18) ^ old) >> 27);
    uint rot = (uint)(old >> 59);
    return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
}

inline float pcg_f32(thread Pcg32& r) {
    return (float)(pcg_next(r) >> 8) * (1.0f / 16777216.0f);
}

inline ulong mix64(ulong z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL;
    return z ^ (z >> 31);
}

inline Pcg32 pcg_for_pixel_sample(ulong pixel, ulong sample) {
    ulong seed = mix64(pixel * 0x9e3779b97f4a7c15UL ^ sample);
    ulong stream = mix64(sample) | 1UL;
    Pcg32 r;
    r.state = 0;
    r.inc = (stream << 1) | 1UL;
    pcg_next(r);
    r.state += seed;
    pcg_next(r);
    return r;
}

// ---- traversal ----

// Reciprocal that never produces INF (fast-math-safe slab tests).
inline float3 safe_inv(float3 d) {
    float3 a = fabs(d);
    float sx = d.x >= 0.0f ? 1.0f : -1.0f;
    float sy = d.y >= 0.0f ? 1.0f : -1.0f;
    float sz = d.z >= 0.0f ? 1.0f : -1.0f;
    return float3(sx / max(a.x, 1e-12f), sy / max(a.y, 1e-12f), sz / max(a.z, 1e-12f));
}

inline bool aabb_hit(device const BvhNodeG& node, float3 o, float3 inv_d,
                     float t_max, thread float& t_near_out) {
    float3 mn = float3(node.mn[0], node.mn[1], node.mn[2]);
    float3 mx = float3(node.mx[0], node.mx[1], node.mx[2]);
    float3 t0 = (mn - o) * inv_d;
    float3 t1 = (mx - o) * inv_d;
    float3 tsm = min(t0, t1);
    float3 tbg = max(t0, t1);
    float t_near = max(max(tsm.x, tsm.y), tsm.z);
    float t_far = min(min(tbg.x, tbg.y), tbg.z);
    if (t_near <= t_far && t_far > 0.0f && t_near < t_max) {
        t_near_out = max(t_near, 0.0f);
        return true;
    }
    return false;
}

struct PtHit {
    bool  hit;
    float t;
    float3 p;
    float3 n;
    uint  material;
    bool  front;
};

// Möller–Trumbore; returns t (parametric along dir) with barycentrics.
inline bool tri_hit(float3 o, float3 d, float3 v0, float3 v1, float3 v2,
                    float t_max, thread float& t_out,
                    thread float& u_out, thread float& v_out) {
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 pv = cross(d, e2);
    float det = dot(e1, pv);
    if (fabs(det) < 1e-12f) return false;
    float invd = 1.0f / det;
    float3 tv = o - v0;
    float uu = dot(tv, pv) * invd;
    if (uu < 0.0f || uu > 1.0f) return false;
    float3 qv = cross(tv, e1);
    float vv = dot(d, qv) * invd;
    if (vv < 0.0f || uu + vv > 1.0f) return false;
    float t = dot(e2, qv) * invd;
    if (t <= 1e-6f || t >= t_max) return false;
    t_out = t;
    u_out = uu;
    v_out = vv;
    return true;
}

inline float3 fetch_v3(device const float* buf, uint index3) {
    return float3(buf[index3 * 3u], buf[index3 * 3u + 1u], buf[index3 * 3u + 2u]);
}

// Closest hit within one instance's BLAS. t stays parametric along the
// world direction (the instance transform is applied without normalizing).
inline bool instance_hit(thread const PtScene& s, uint inst_id,
                         float3 wo_pos, float3 wd, float t_max,
                         thread float& t_out, thread float3& n_out) {
    device const InstanceG& inst = s.instances[inst_id];
    device const MeshInfoG& mesh = s.mesh_infos[inst.mesh_id];
    Affine inv = load_affine(inst.inv);
    float3 o = xf_point(inv, wo_pos);
    float3 d = xf_vec(inv, wd);
    float3 inv_d = safe_inv(d);

    float best_t = t_max;
    uint best_tri = 0xFFFFFFFFu;
    float best_u = 0.0f;
    float best_v = 0.0f;

    uint stack[48];
    int sp = 0;
    stack[sp++] = 0u;
    while (sp > 0) {
        device const BvhNodeG& node = s.blas[mesh.node_offset + stack[--sp]];
        float tn;
        if (!aabb_hit(node, o, inv_d, best_t, tn)) continue;
        if (node.count > 0u) {
            for (uint i = 0u; i < node.count; i++) {
                uint tri = node.left_or_first + i;
                uint base = mesh.index_offset + tri * 3u;
                float3 v0 = fetch_v3(s.vertices, mesh.vertex_offset + s.tri_indices[base]);
                float3 v1 = fetch_v3(s.vertices, mesh.vertex_offset + s.tri_indices[base + 1u]);
                float3 v2 = fetch_v3(s.vertices, mesh.vertex_offset + s.tri_indices[base + 2u]);
                float t, uu, vv;
                if (tri_hit(o, d, v0, v1, v2, best_t, t, uu, vv)) {
                    best_t = t;
                    best_tri = tri;
                    best_u = uu;
                    best_v = vv;
                }
            }
        } else {
            // Near-first ordering: essential for early best_t pruning on
            // deep BVHs (unordered DFS visits far more nodes).
            uint l = node.left_or_first;
            uint r = l + 1u;
            float tl, tr;
            bool hl = aabb_hit(s.blas[mesh.node_offset + l], o, inv_d, best_t, tl);
            bool hr = aabb_hit(s.blas[mesh.node_offset + r], o, inv_d, best_t, tr);
            if (hl && hr) {
                uint near_c = (tl <= tr) ? l : r;
                uint far_c = (tl <= tr) ? r : l;
                stack[sp++] = far_c;
                stack[sp++] = near_c;
            } else if (hl) {
                stack[sp++] = l;
            } else if (hr) {
                stack[sp++] = r;
            }
        }
    }
    if (best_tri == 0xFFFFFFFFu) return false;

    // Normal: interpolated when present, geometric otherwise.
    uint base = mesh.index_offset + best_tri * 3u;
    uint i0 = s.tri_indices[base];
    uint i1 = s.tri_indices[base + 1u];
    uint i2 = s.tri_indices[base + 2u];
    float3 nl;
    if (mesh.has_normals != 0u) {
        float3 n0 = fetch_v3(s.normals, mesh.vertex_offset + i0);
        float3 n1 = fetch_v3(s.normals, mesh.vertex_offset + i1);
        float3 n2 = fetch_v3(s.normals, mesh.vertex_offset + i2);
        float w = 1.0f - best_u - best_v;
        nl = normalize_cpu(n0 * w + n1 * best_u + n2 * best_v);
    } else {
        float3 v0 = fetch_v3(s.vertices, mesh.vertex_offset + i0);
        float3 v1 = fetch_v3(s.vertices, mesh.vertex_offset + i1);
        float3 v2 = fetch_v3(s.vertices, mesh.vertex_offset + i2);
        nl = normalize_cpu(cross(v1 - v0, v2 - v0));
    }
    t_out = best_t;
    n_out = normalize_cpu(xf_normal(inv, nl));
    return true;
}

// Any triangle hit within one instance before t_limit (shadow rays).
inline bool instance_occludes(thread const PtScene& s, uint inst_id,
                              float3 wo_pos, float3 wd, float t_limit) {
    device const InstanceG& inst = s.instances[inst_id];
    device const MeshInfoG& mesh = s.mesh_infos[inst.mesh_id];
    Affine inv = load_affine(inst.inv);
    float3 o = xf_point(inv, wo_pos);
    float3 d = xf_vec(inv, wd);
    float3 inv_d = safe_inv(d);

    uint stack[48];
    int sp = 0;
    stack[sp++] = 0u;
    while (sp > 0) {
        device const BvhNodeG& node = s.blas[mesh.node_offset + stack[--sp]];
        float tn;
        if (!aabb_hit(node, o, inv_d, t_limit, tn)) continue;
        if (node.count > 0u) {
            for (uint i = 0u; i < node.count; i++) {
                uint tri = node.left_or_first + i;
                uint base = mesh.index_offset + tri * 3u;
                float3 v0 = fetch_v3(s.vertices, mesh.vertex_offset + s.tri_indices[base]);
                float3 v1 = fetch_v3(s.vertices, mesh.vertex_offset + s.tri_indices[base + 1u]);
                float3 v2 = fetch_v3(s.vertices, mesh.vertex_offset + s.tri_indices[base + 2u]);
                float t, uu, vv;
                if (tri_hit(o, d, v0, v1, v2, t_limit, t, uu, vv)) return true;
            }
        } else {
            stack[sp++] = node.left_or_first;
            stack[sp++] = node.left_or_first + 1u;
        }
    }
    return false;
}

// Closest hit across quadrics + instanced meshes. World directions are
// unit length, so quadric euclidean t and mesh parametric t agree.
inline PtHit pt_trace_scene(thread const PtScene& s, float3 o, float3 d) {
    PtHit best;
    best.hit = false;
    best.t = PT_BIG;
    best.p = float3(0.0f);
    best.n = float3(0.0f);
    best.material = 0u;
    best.front = true;

    for (uint i = 0u; i < s.u->object_count; i++) {
        Hit h = isect_object(s.objects[i], o, d);
        if (h.valid && h.t < best.t) {
            best.hit = true;
            best.t = h.t;
            best.p = h.p;
            best.n = h.n;
            best.material = s.object_materials[i];
            best.front = h.front;
        }
    }

    if (s.u->instance_count > 0u) {
        float3 inv_d = safe_inv(d);
        uint stack[40];
        int sp = 0;
        stack[sp++] = 0u;
        while (sp > 0) {
            device const BvhNodeG& node = s.tlas[stack[--sp]];
            float tn;
            if (!aabb_hit(node, o, inv_d, best.t, tn)) continue;
            if (node.count > 0u) {
                for (uint i = 0u; i < node.count; i++) {
                    uint inst_id = node.left_or_first + i;
                    float t;
                    float3 n;
                    if (instance_hit(s, inst_id, o, d, best.t, t, n)) {
                        best.hit = true;
                        best.t = t;
                        best.p = o + d * t;
                        // Unflipped mesh normal: side falls out of the dot.
                        best.front = dot(n, d) < 0.0f;
                        best.n = n;
                        best.material = s.instances[inst_id].material_id;
                    }
                }
            } else {
                uint l = node.left_or_first;
                uint r = l + 1u;
                float tl, tr;
                bool hl = aabb_hit(s.tlas[l], o, inv_d, best.t, tl);
                bool hr = aabb_hit(s.tlas[r], o, inv_d, best.t, tr);
                if (hl && hr) {
                    uint near_c = (tl <= tr) ? l : r;
                    uint far_c = (tl <= tr) ? r : l;
                    stack[sp++] = far_c;
                    stack[sp++] = near_c;
                } else if (hl) {
                    stack[sp++] = l;
                } else if (hr) {
                    stack[sp++] = r;
                }
            }
        }
    }
    return best;
}

// Any hit strictly before t_limit (shadow rays; dir unit length).
inline bool pt_occluded(thread const PtScene& s, float3 p, float3 n,
                        float3 dir, float dist) {
    float3 o = p + n * PT_RAY_OFFSET;
    float limit = dist - 1e-3f;
    for (uint i = 0u; i < s.u->object_count; i++) {
        Hit h = isect_object(s.objects[i], o, dir);
        if (h.valid && h.t < limit) return true;
    }
    if (s.u->instance_count > 0u) {
        float3 inv_d = safe_inv(dir);
        uint stack[40];
        int sp = 0;
        stack[sp++] = 0u;
        while (sp > 0) {
            device const BvhNodeG& node = s.tlas[stack[--sp]];
            float tn;
            if (!aabb_hit(node, o, inv_d, limit, tn)) continue;
            if (node.count > 0u) {
                for (uint i = 0u; i < node.count; i++) {
                    if (instance_occludes(s, node.left_or_first + i, o, dir, limit)) {
                        return true;
                    }
                }
            } else {
                stack[sp++] = node.left_or_first;
                stack[sp++] = node.left_or_first + 1u;
            }
        }
    }
    return false;
}

// ---- BSDF lobes (mirror of pt/bxdf.rs: Oren-Nayar, GGX/VNDF with
//      height-correlated Smith, clearcoat, fuzz, rough glass R/T) ----

inline float3 m3(thread const float* a) { return float3(a[0], a[1], a[2]); }
inline float3 m3(device const float* a) { return float3(a[0], a[1], a[2]); }
inline float3 c3(constant const float* a) { return float3(a[0], a[1], a[2]); }

inline float3 schlick3(float3 f0, float3 f90, float c) {
    float m = pow(clamp(1.0f - c, 0.0f, 1.0f), 5.0f);
    return f0 + (f90 - f0) * m;
}

inline float fresnel_dielectric(float ci, float eta) {
    ci = clamp(ci, 0.0f, 1.0f);
    float s2 = (1.0f - ci * ci) / (eta * eta);
    if (s2 >= 1.0f) return 1.0f;
    float ct = sqrt(1.0f - s2);
    float rp = (eta * ci - ct) / (eta * ci + ct);
    float rs = (ci - eta * ct) / (ci + eta * ct);
    return 0.5f * (rp * rp + rs * rs);
}

inline float ggx_d(float3 h, float alpha) {
    if (h.z <= 0.0f) return 0.0f;
    float a2 = alpha * alpha;
    float d = h.z * h.z * (a2 - 1.0f) + 1.0f;
    return a2 / (PT_PI * d * d);
}

inline float ggx_lambda(float3 w, float alpha) {
    float c2 = w.z * w.z;
    if (c2 <= 0.0f) return 0.0f;
    float t2 = max(1.0f - c2, 0.0f) / c2;
    return (sqrt(1.0f + alpha * alpha * t2) - 1.0f) * 0.5f;
}

inline float ggx_g1(float3 w, float alpha) { return 1.0f / (1.0f + ggx_lambda(w, alpha)); }
inline float ggx_g2(float3 wo, float3 wi, float alpha) {
    return 1.0f / (1.0f + ggx_lambda(wo, alpha) + ggx_lambda(wi, alpha));
}

inline float3 ggx_sample_vndf(float3 wo, float alpha, float u1, float u2) {
    float3 v = normalize_cpu(float3(alpha * wo.x, alpha * wo.y, wo.z));
    float lensq = v.x * v.x + v.y * v.y;
    float3 t1 = lensq > 1e-12f ? float3(-v.y, v.x, 0.0f) / sqrt(lensq)
                               : float3(1.0f, 0.0f, 0.0f);
    float3 t2 = cross(v, t1);
    float r = sqrt(u1);
    float phi = 2.0f * PT_PI * u2;
    float p1 = r * cos(phi);
    float p2 = r * sin(phi);
    float sfac = 0.5f * (1.0f + v.z);
    p2 = (1.0f - sfac) * sqrt(max(1.0f - p1 * p1, 0.0f)) + sfac * p2;
    float3 nh = t1 * p1 + t2 * p2 + v * sqrt(max(1.0f - p1 * p1 - p2 * p2, 0.0f));
    return normalize_cpu(float3(alpha * nh.x, alpha * nh.y, max(nh.z, 1e-9f)));
}

inline float ggx_pdf_h(float3 wo, float3 h, float alpha) {
    float odh = dot(wo, h);
    if (odh <= 0.0f || wo.z <= 0.0f) return 0.0f;
    return ggx_g1(wo, alpha) * odh * ggx_d(h, alpha) / wo.z;
}

inline float3 eval_diffuse(PtMaterial m, float3 wo, float3 wi) {
    float s2 = m.diffuse_sigma * m.diffuse_sigma;
    float a = 1.0f - s2 / (2.0f * (s2 + 0.33f));
    float b = 0.45f * s2 / (s2 + 0.09f);
    float so = sqrt(max(1.0f - wo.z * wo.z, 0.0f));
    float si = sqrt(max(1.0f - wi.z * wi.z, 0.0f));
    float cd = (so > 1e-6f && si > 1e-6f)
        ? clamp((wo.x * wi.x + wo.y * wi.y) / (so * si), -1.0f, 1.0f)
        : 0.0f;
    float sa, tb;
    if (wo.z < wi.z) { sa = so; tb = si / max(wi.z, 1e-6f); }
    else { sa = si; tb = so / max(wo.z, 1e-6f); }
    return m3(m.diffuse_color) * (m.diffuse_gain * m.under_scale / PT_PI)
        * (a + b * max(cd, 0.0f) * sa * tb);
}

inline float3 eval_fuzz(PtMaterial m, float3 wo, float3 wi) {
    float rim = 0.5f * (pow(1.0f - wi.z, 4.0f) + pow(1.0f - wo.z, 4.0f));
    return m3(m.fuzz_color) * (m.fuzz_gain / PT_PI) * rim;
}

inline float3 eval_spec_lobe(float3 f0, float3 f90, float alpha, float3 wo, float3 wi) {
    float3 h = normalize_cpu(wo + wi);
    float d = ggx_d(h, alpha);
    float g = ggx_g2(wo, wi, alpha);
    float3 fr = schlick3(f0, f90, max(dot(wo, h), 0.0f));
    return fr * (d * g / max(4.0f * wo.z * wi.z, 1e-9f));
}

inline float spec_pdf(float3 wo, float3 wi, float alpha) {
    float3 h = normalize_cpu(wo + wi);
    float odh = max(dot(wo, h), 1e-9f);
    return ggx_pdf_h(wo, h, alpha) / (4.0f * odh);
}

// Glass transmission (PBRT-3 microfacet transmission, radiance transport).
inline void glass_transmit(PtMaterial m, float3 wo, float3 wi, float eta,
                           thread float3& f_out, thread float& pdf_out) {
    f_out = float3(0.0f);
    pdf_out = 0.0f;
    float alpha = m.glass_alpha;
    float3 h = normalize_cpu(wo + wi * eta);
    if (h.z < 0.0f) h = -h;
    float odh = dot(wo, h);
    float idh = dot(wi, h);
    if (odh <= 0.0f || idh >= 0.0f) return;
    float fres = fresnel_dielectric(odh, eta);
    float d = ggx_d(h, alpha);
    float g = ggx_g2(wo, float3(wi.x, wi.y, -wi.z), alpha);
    float sq = odh + eta * idh;
    if (fabs(sq) < 1e-9f) return;
    f_out = m3(m.refr_color)
        * (m.glass_gain * m.under_scale * (1.0f - fres) * d * g
           * fabs(idh * odh / (wi.z * wo.z)) / (sq * sq));
    float dwh = fabs(eta * eta * idh) / (sq * sq);
    pdf_out = ggx_pdf_h(wo, h, alpha) * dwh;
}

// Full composite eval + pdf; wo.z > 0, wi either hemisphere.
inline void bsdf_eval_pdf(PtMaterial m, float3 wo, float3 wi, float eta,
                          thread float3& f_out, thread float& pdf_out) {
    f_out = float3(0.0f);
    pdf_out = 0.0f;
    float wd = m.weights[0], ws = m.weights[1], wc = m.weights[2],
          wf = m.weights[3], wg = m.weights[4];
    if (wd + ws + wc + wf + wg <= 0.0f) return;

    if (wi.z > 0.0f) {
        float cos_pdf = wi.z / PT_PI;
        if (wd > 0.0f) { f_out += eval_diffuse(m, wo, wi); pdf_out += wd * cos_pdf; }
        if (wf > 0.0f) { f_out += eval_fuzz(m, wo, wi); pdf_out += wf * cos_pdf; }
        if (ws > 0.0f) {
            f_out += eval_spec_lobe(m3(m.spec_f0), m3(m.spec_f90), m.spec_alpha, wo, wi);
            pdf_out += ws * spec_pdf(wo, wi, m.spec_alpha);
        }
        if (wc > 0.0f) {
            float3 cf0 = float3(0.04f) * m.coat_gain;
            f_out += eval_spec_lobe(cf0, float3(m.coat_gain), m.coat_alpha, wo, wi);
            pdf_out += wc * spec_pdf(wo, wi, m.coat_alpha);
        }
        if (wg > 0.0f) {
            float3 h = normalize_cpu(wo + wi);
            float fres = fresnel_dielectric(max(dot(wo, h), 0.0f), eta);
            float d = ggx_d(h, m.glass_alpha);
            float g = ggx_g2(wo, wi, m.glass_alpha);
            f_out += float3(m.glass_gain * m.under_scale * fres * d * g
                / max(4.0f * wo.z * wi.z, 1e-9f));
            pdf_out += wg * fres * spec_pdf(wo, wi, m.glass_alpha);
        }
    } else if (wg > 0.0f) {
        float3 f;
        float pl;
        glass_transmit(m, wo, wi, eta, f, pl);
        float3 h = normalize_cpu(wo + wi * eta);
        if (h.z < 0.0f) h = -h;
        float fres = fresnel_dielectric(max(dot(wo, h), 0.0f), eta);
        f_out = f;
        pdf_out = wg * (1.0f - fres) * pl;
    }
}

inline float3 cosine_sample_local(thread Pcg32& rng) {
    float u = pcg_f32(rng);
    float v = pcg_f32(rng);
    float r = sqrt(u);
    float phi = 2.0f * PT_PI * v;
    return float3(r * cos(phi), r * sin(phi), sqrt(max(1.0f - u, 0.0f)));
}

// Composite sample; returns false on rejection.
inline bool bsdf_sample(PtMaterial m, float3 wo, float eta, thread Pcg32& rng,
                        thread float3& wi_out, thread float3& f_out,
                        thread float& pdf_out, thread bool& transmitted) {
    float wd = m.weights[0], ws = m.weights[1], wc = m.weights[2],
          wf = m.weights[3], wg = m.weights[4];
    if (wd + ws + wc + wf + wg <= 0.0f) return false;
    transmitted = false;
    float pick = pcg_f32(rng);
    float3 wi;
    if (pick < wd + wf) {
        wi = cosine_sample_local(rng);
    } else if (pick < wd + wf + ws) {
        float u1 = pcg_f32(rng);
        float u2 = pcg_f32(rng);
        float3 h = ggx_sample_vndf(wo, m.spec_alpha, u1, u2);
        wi = reflect(-wo, h);
        if (wi.z <= 0.0f) return false;
    } else if (pick < wd + wf + ws + wc) {
        float u1 = pcg_f32(rng);
        float u2 = pcg_f32(rng);
        float3 h = ggx_sample_vndf(wo, m.coat_alpha, u1, u2);
        wi = reflect(-wo, h);
        if (wi.z <= 0.0f) return false;
    } else if (wg > 0.0f) {
        float u1 = pcg_f32(rng);
        float u2 = pcg_f32(rng);
        float3 h = ggx_sample_vndf(wo, m.glass_alpha, u1, u2);
        float fres = fresnel_dielectric(max(dot(wo, h), 0.0f), eta);
        if (pcg_f32(rng) < fres) {
            wi = reflect(-wo, h);
            if (wi.z <= 0.0f) return false;
        } else {
            float ci = dot(wo, h);
            float s2 = (1.0f - ci * ci) / (eta * eta);
            if (s2 >= 1.0f) return false;
            float ct = sqrt(1.0f - s2);
            wi = normalize_cpu((-wo) / eta + h * (ci / eta - ct));
            if (wi.z >= 0.0f) return false;
            transmitted = true;
        }
    } else {
        return false;
    }
    float3 f;
    float pdf;
    bsdf_eval_pdf(m, wo, wi, eta, f, pdf);
    if (pdf <= 0.0f) return false;
    wi_out = wi;
    f_out = f;
    pdf_out = pdf;
    return true;
}

// Local shading frame.
struct FrameL { float3 t; float3 b; float3 n; };
inline FrameL frame_of(float3 n) {
    float3 t0 = fabs(n.x) > 0.9f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    FrameL f;
    f.n = n;
    f.t = normalize_cpu(cross(n, t0));
    f.b = cross(n, f.t);
    return f;
}
inline float3 to_local(FrameL f, float3 w) {
    return float3(dot(w, f.t), dot(w, f.b), dot(w, f.n));
}
inline float3 to_world(FrameL f, float3 w) {
    return f.t * w.x + f.b * w.y + f.n * w.z;
}

// ---- lights + dome ------------------------------------------------------

inline float power_heuristic(float a, float b) {
    float a2 = a * a;
    float b2 = b * b;
    return (a2 + b2 <= 0.0f) ? 0.0f : a2 / (a2 + b2);
}

// Environment map helpers (lat-long, y-up; mirrors scene/envmap.rs).
inline float2 env_uv_of(float3 d) {
    float u = atan2(d.x, -d.z) / (2.0f * PT_PI);
    u = u - floor(u);
    float v = clamp(acos(clamp(d.y, -1.0f, 1.0f)) / PT_PI, 0.0f, 1.0f);
    return float2(u, v);
}

inline float3 env_eval(thread const PtScene& s, float3 d) {
    constant PtUniforms& u = *s.u;
    if (u.env_width == 0u) return float3(1.0f);
    float2 uv = env_uv_of(d);
    uint x = min((uint)(uv.x * (float)u.env_width), u.env_width - 1u);
    uint y = min((uint)(uv.y * (float)u.env_height), u.env_height - 1u);
    uint i = (y * u.env_width + x) * 3u;
    return float3(s.env_pixels[i], s.env_pixels[i + 1u], s.env_pixels[i + 2u]);
}

inline float env_pdf(thread const PtScene& s, float3 d) {
    constant PtUniforms& u = *s.u;
    if (u.env_width == 0u) return 1.0f / (4.0f * PT_PI);
    float2 uv = env_uv_of(d);
    uint x = min((uint)(uv.x * (float)u.env_width), u.env_width - 1u);
    uint y = min((uint)(uv.y * (float)u.env_height), u.env_height - 1u);
    float st = sin(PT_PI * ((float)y + 0.5f) / (float)u.env_height);
    if (st < 1e-9f) return 0.0f;
    uint i = (y * u.env_width + x) * 3u;
    float l = 0.2126f * s.env_pixels[i] + 0.7152f * s.env_pixels[i + 1u]
        + 0.0722f * s.env_pixels[i + 2u];
    return l * st * (float)u.env_width * (float)u.env_height
        / (max(u.env_total, 1e-12f) * 2.0f * PT_PI * PT_PI * st);
}

inline uint upper_bound(device const float* cdf, uint count, float target) {
    uint lo = 0u;
    uint hi = count;
    while (lo < hi) {
        uint mid = (lo + hi) / 2u;
        if (cdf[mid] <= target) lo = mid + 1u; else hi = mid;
    }
    return lo == 0u ? 0u : lo - 1u;
}

inline void env_sample(thread const PtScene& s, thread Pcg32& rng,
                       thread float3& dir, thread float3& rad, thread float& pdf) {
    constant PtUniforms& u = *s.u;
    if (u.env_width == 0u) {
        // Constant dome: uniform sphere.
        float uu = pcg_f32(rng);
        float vv = pcg_f32(rng);
        float z = 1.0f - 2.0f * uu;
        float r = sqrt(max(1.0f - z * z, 0.0f));
        float phi = 2.0f * PT_PI * vv;
        dir = float3(r * cos(phi), r * sin(phi), z);
        rad = float3(1.0f);
        pdf = 1.0f / (4.0f * PT_PI);
        return;
    }
    float t1 = pcg_f32(rng) * u.env_total;
    uint y = min(upper_bound(s.env_marginal, u.env_height + 1u, t1), u.env_height - 1u);
    float row_lo = s.env_marginal[y];
    float row_hi = s.env_marginal[y + 1u];
    float t2 = pcg_f32(rng) * max(row_hi - row_lo, 1e-12f);
    uint base = y * (u.env_width + 1u);
    uint x = min(upper_bound(s.env_conditional + base, u.env_width + 1u, t2), u.env_width - 1u);
    float uu = ((float)x + 0.5f) / (float)u.env_width;
    float vv = ((float)y + 0.5f) / (float)u.env_height;
    float phi = uu * 2.0f * PT_PI;
    float theta = vv * PT_PI;
    float st = sin(theta);
    dir = float3(st * sin(phi), cos(theta), -st * cos(phi));
    uint i = (y * u.env_width + x) * 3u;
    rad = float3(s.env_pixels[i], s.env_pixels[i + 1u], s.env_pixels[i + 2u]);
    pdf = env_pdf(s, dir);
}

// pdf of an area light producing the direction toward hit_p (for MIS).
inline float light_pdf_solid_angle(PtLight l, float3 origin, float3 hit_p) {
    if (l.kind == 2u || l.kind == 4u) {
        float3 d = hit_p - origin;
        float dist2 = dot(d, d);
        float3 ln = m3(l.normal);
        float cl = fabs(dot(normalize_cpu(d), ln));
        if (cl < 1e-9f || l.area <= 0.0f) return 0.0f;
        return dist2 / (cl * l.area);
    }
    if (l.kind == 3u) {
        float3 c = m3(l.a);
        float dist2 = dot(c - origin, c - origin);
        float s2 = min(l.area * l.area / dist2, 1.0f);
        float cm = sqrt(max(1.0f - s2, 0.0f));
        float sa = 2.0f * PT_PI * (1.0f - cm);
        return sa < 1e-12f ? 0.0f : 1.0f / sa;
    }
    return 0.0f;
}

// Fractional visibility (presence cutouts attenuate; opaque kills).
inline float pt_visibility(thread const PtScene& s, float3 p, float3 n,
                           float3 dir, float dist) {
    float3 origin = p + n * PT_RAY_OFFSET;
    float remaining = dist - 1e-3f;
    float vis = 1.0f;
    for (int i = 0; i < 16; i++) {
        PtHit h = pt_trace_scene(s, origin, dir);
        if (!h.hit || h.t >= remaining) return vis;
        float presence = clamp(s.materials[h.material].presence, 0.0f, 1.0f);
        if (presence >= 1.0f) return 0.0f;
        vis *= 1.0f - presence;
        if (vis < 1e-4f) return 0.0f;
        origin = h.p + dir * PT_RAY_OFFSET;
        remaining -= h.t + PT_RAY_OFFSET;
    }
    return vis;
}

inline float3 sample_one_light(thread const PtScene& s, PtLight l, float3 p,
                               FrameL frame, float3 wo_l, PtMaterial m,
                               float eta, thread Pcg32& rng) {
    float3 n = frame.n;
    float3 rad = m3(l.radiance);

    if (l.kind == 0u) {                       // point
        float3 to_l = m3(l.a) - p;
        float dist2 = max(dot(to_l, to_l), 1e-12f);
        float dist = sqrt(dist2);
        float3 wi = to_l / dist;
        float3 wi_l = to_local(frame, wi);
        if (wi_l.z <= 0.0f) return float3(0.0f);
        float3 f;
        float pdf;
        bsdf_eval_pdf(m, wo_l, wi_l, eta, f, pdf);
        if (max(max(f.x, f.y), f.z) <= 0.0f) return float3(0.0f);
        float vis = pt_visibility(s, p, n, wi, dist);
        return f * rad * (max(wi_l.z, 0.0f) * vis / dist2);
    }
    if (l.kind == 1u) {                       // distant (soft when area>0)
        float3 base = -m3(l.a);
        float3 wi = base;
        if (l.area > 1e-5f) {
            float cm = cos(l.area);
            float uu = pcg_f32(rng);
            float vv = pcg_f32(rng);
            float ct = 1.0f - uu * (1.0f - cm);
            float st = sqrt(max(1.0f - ct * ct, 0.0f));
            float phi = 2.0f * PT_PI * vv;
            FrameL lf = frame_of(base);
            wi = to_world(lf, float3(st * cos(phi), st * sin(phi), ct));
        }
        float3 wi_l = to_local(frame, wi);
        if (wi_l.z <= 0.0f) return float3(0.0f);
        float3 f;
        float pdf;
        bsdf_eval_pdf(m, wo_l, wi_l, eta, f, pdf);
        if (max(max(f.x, f.y), f.z) <= 0.0f) return float3(0.0f);
        float vis = pt_visibility(s, p, n, wi, PT_BIG);
        return f * rad * (max(wi_l.z, 0.0f) * vis);
    }
    if (l.kind == 2u || l.kind == 4u) {       // rect / disk
        float uu = pcg_f32(rng);
        float vv = pcg_f32(rng);
        float3 sp;
        if (l.kind == 2u) {
            sp = m3(l.a) + m3(l.e1) * uu + m3(l.e2) * vv;
        } else {
            float r = sqrt(uu);
            float phi = 2.0f * PT_PI * vv;
            sp = m3(l.a) + m3(l.e1) * (r * cos(phi)) + m3(l.e2) * (r * sin(phi));
        }
        float3 to_l = sp - p;
        float dist2 = max(dot(to_l, to_l), 1e-12f);
        float dist = sqrt(dist2);
        float3 wi = to_l / dist;
        float3 wi_l = to_local(frame, wi);
        if (wi_l.z <= 0.0f) return float3(0.0f);
        float cl = fabs(dot(wi, m3(l.normal)));
        if (cl < 1e-9f || l.area <= 0.0f) return float3(0.0f);
        float3 f;
        float bp;
        bsdf_eval_pdf(m, wo_l, wi_l, eta, f, bp);
        if (max(max(f.x, f.y), f.z) <= 0.0f) return float3(0.0f);
        float vis = pt_visibility(s, p, n, wi, dist);
        if (vis <= 0.0f) return float3(0.0f);
        float pdf_sa = dist2 / (cl * l.area);
        float w = power_heuristic(pdf_sa, bp);
        return f * rad * (max(wi_l.z, 0.0f) * vis / pdf_sa) * w;
    }
    if (l.kind == 3u) {                       // sphere area: visible cone
        float3 c = m3(l.a);
        float3 to_c = c - p;
        float dist2 = dot(to_c, to_c);
        float radius = l.area;
        if (dist2 <= radius * radius * 1.0001f) return float3(0.0f);
        float s2 = min(radius * radius / dist2, 1.0f);
        float cm = sqrt(max(1.0f - s2, 0.0f));
        float uu = pcg_f32(rng);
        float vv = pcg_f32(rng);
        float ct = 1.0f - uu * (1.0f - cm);
        float st = sqrt(max(1.0f - ct * ct, 0.0f));
        float phi = 2.0f * PT_PI * vv;
        FrameL lf = frame_of(normalize_cpu(to_c));
        float3 wi = to_world(lf, float3(st * cos(phi), st * sin(phi), ct));
        float3 wi_l = to_local(frame, wi);
        if (wi_l.z <= 0.0f) return float3(0.0f);
        float3 f;
        float bp;
        bsdf_eval_pdf(m, wo_l, wi_l, eta, f, bp);
        if (max(max(f.x, f.y), f.z) <= 0.0f) return float3(0.0f);
        float dist = sqrt(dist2) * ct
            - sqrt(max(radius * radius - dist2 * st * st, 0.0f));
        float vis = pt_visibility(s, p, n, wi, dist);
        if (vis <= 0.0f) return float3(0.0f);
        float pdf_sa = 1.0f / max(2.0f * PT_PI * (1.0f - cm), 1e-12f);
        float w = power_heuristic(pdf_sa, bp);
        return f * rad * (max(wi_l.z, 0.0f) * vis / pdf_sa) * w;
    }
    // dome
    float3 wi, er;
    float pdf_sa;
    env_sample(s, rng, wi, er, pdf_sa);
    if (pdf_sa <= 0.0f) return float3(0.0f);
    float3 wi_l = to_local(frame, wi);
    if (wi_l.z <= 0.0f) return float3(0.0f);
    float3 f;
    float bp;
    bsdf_eval_pdf(m, wo_l, wi_l, eta, f, bp);
    if (max(max(f.x, f.y), f.z) <= 0.0f) return float3(0.0f);
    float vis = pt_visibility(s, p, n, wi, PT_BIG);
    if (vis <= 0.0f) return float3(0.0f);
    float w = power_heuristic(pdf_sa, bp);
    return f * (er * rad) * (max(wi_l.z, 0.0f) * vis / pdf_sa) * w;
}

// ---- the path integrator (mirror of pt::trace) --------------------------

inline float3 pt_li(thread const PtScene& s, float3 origin, float3 dir,
                    thread Pcg32& rng) {
    constant PtUniforms& u = *s.u;
    float3 l = float3(0.0f);
    float3 beta = float3(1.0f);
    float prev_pdf = 0.0f;
    float3 prev_origin = origin;
    bool from_camera = true;
    uint presence_skips = 0u;
    float num_lights = (float)u.light_count;

    uint depth = 0u;
    while (depth < u.max_bounces) {
        PtHit h = pt_trace_scene(s, origin, dir);
        if (!h.hit) {
            if (u.dome_index != 0xFFFFFFFFu) {
                float3 d = normalize_cpu(dir);
                float w = 1.0f;
                if (!from_camera) {
                    float pl = env_pdf(s, d) / num_lights;
                    w = power_heuristic(prev_pdf, pl);
                }
                float3 dome_rad = env_eval(s, d) * m3(s.lights[u.dome_index].radiance);
                l += beta * dome_rad * w;
            } else {
                l += beta * c3(u.background);
            }
            break;
        }
        PtMaterial m = s.materials[h.material];
        float3 wo = -normalize_cpu(dir);

        float presence = clamp(m.presence, 0.0f, 1.0f);
        if (presence < 1.0f && pcg_f32(rng) >= presence && presence_skips < 16u) {
            presence_skips++;
            origin = h.p + normalize_cpu(dir) * PT_RAY_OFFSET;
            continue;
        }

        bool entering = h.front;
        float3 n = dot(h.n, wo) >= 0.0f ? h.n : -h.n;
        float eta = entering ? m.glass_ior : 1.0f / max(m.glass_ior, 1e-6f);

        float3 em = m3(m.emission);
        if (max(max(em.x, em.y), em.z) > 0.0f) {
            float w = 1.0f;
            if (!from_camera && m.area_light != 0xFFFFFFFFu) {
                float pl = light_pdf_solid_angle(s.lights[m.area_light], prev_origin, h.p)
                    / num_lights;
                w = power_heuristic(prev_pdf, pl);
            }
            l += beta * em * w;
        }

        FrameL frame = frame_of(n);
        float3 wo_l = to_local(frame, wo);
        float wsum = m.weights[0] + m.weights[1] + m.weights[2] + m.weights[3]
            + m.weights[4];

        if (u.light_count > 0u && wsum > 0.0f) {
            uint li = min((uint)(pcg_f32(rng) * num_lights), u.light_count - 1u);
            float3 c = sample_one_light(s, s.lights[li], h.p, frame, wo_l, m, eta, rng);
            l += beta * c * num_lights;
        }

        float3 wi_l, f;
        float pdf;
        bool transmitted;
        if (!bsdf_sample(m, wo_l, eta, rng, wi_l, f, pdf, transmitted)) break;
        float3 wi = to_world(frame, wi_l);
        beta *= f * (fabs(wi_l.z) / pdf);
        prev_pdf = pdf;
        prev_origin = h.p;
        from_camera = false;

        float off = transmitted ? -PT_RAY_OFFSET : PT_RAY_OFFSET;
        origin = h.p + n * off;
        dir = wi;
        depth++;

        if (depth >= u.rr_start) {
            float q = min(max(max(beta.x, beta.y), beta.z), 0.95f);
            if (q <= 0.0f || pcg_f32(rng) > q) break;
            beta /= q;
        }
    }
    return l;
}

// ---- entry: one thread per pixel, one sample per dispatch ---------------

kernel void render_pt(device const Object*     objects          [[buffer(0)]],
                      device const uint*       object_materials [[buffer(1)]],
                      device const PtMaterial* materials        [[buffer(2)]],
                      device const PtLight*    lights           [[buffer(3)]],
                      device const BvhNodeG*   tlas             [[buffer(4)]],
                      device const InstanceG*  instances        [[buffer(5)]],
                      device const BvhNodeG*   blas             [[buffer(6)]],
                      device const uint*       tri_indices      [[buffer(7)]],
                      device const float*      vertices         [[buffer(8)]],
                      device const float*      normals          [[buffer(9)]],
                      device const MeshInfoG*  mesh_infos       [[buffer(10)]],
                      device const float*      env_pixels       [[buffer(11)]],
                      device const float*      env_marginal     [[buffer(12)]],
                      device const float*      env_conditional  [[buffer(13)]],
                      constant PtUniforms&     u                [[buffer(14)]],
                      device float*            accum            [[buffer(15)]],
                      uint2 gid [[thread_position_in_grid]]) {
    uint py_row = gid.y + u.y_offset;
    if (gid.x >= u.width || py_row >= u.height) return;

    PtScene s;
    s.objects = objects;
    s.object_materials = object_materials;
    s.materials = materials;
    s.lights = lights;
    s.tlas = tlas;
    s.instances = instances;
    s.blas = blas;
    s.tri_indices = tri_indices;
    s.vertices = vertices;
    s.normals = normals;
    s.mesh_infos = mesh_infos;
    s.env_pixels = env_pixels;
    s.env_marginal = env_marginal;
    s.env_conditional = env_conditional;
    s.u = &u;

    float3 eye = c3(u.eye);
    float3 fwd = c3(u.forward);
    float3 rgt = c3(u.right);
    float3 upv = c3(u.up);

    ulong pixel = (ulong)(py_row * u.width + gid.x);
    float3 sum = float3(0.0f);
    for (uint sidx = 0u; sidx < u.sample_count; sidx++) {
        Pcg32 rng = pcg_for_pixel_sample(pixel, (ulong)(u.sample_start + sidx));
        float jx = pcg_f32(rng);
        float jy = pcg_f32(rng);
        float px = (float)gid.x + jx;
        float py = (float)py_row + jy;
        float uu = (px / (float)u.width) * 2.0f - 1.0f;
        float vv = 1.0f - (py / (float)u.height) * 2.0f;
        float3 d = normalize_cpu(fwd + rgt * (uu * u.half_width) + upv * (vv * u.half_height));
        float3 li = pt_li(s, eye, d, rng);
        float lum = 0.2126f * li.x + 0.7152f * li.y + 0.0722f * li.z;
        if (lum > u.firefly_clamp) li *= u.firefly_clamp / lum;
        sum += li;
    }

    uint idx = (py_row * u.width + gid.x) * 4u;
    accum[idx + 0u] += sum.x;
    accum[idx + 1u] += sum.y;
    accum[idx + 2u] += sum.z;
    accum[idx + 3u] += (float)u.sample_count;
}
