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
};

struct PtMaterial {
    float color[3];
    uint  kind;          // 0 matte, 1 plastic, 2 metal
    float emission[3];
    float alpha;         // GGX alpha = roughness^2
    uint  area_light;    // light index or 0xFFFFFFFF
    float p_spec;
    float pad[2];
};

struct PtLight {
    uint  kind;          // 0 point, 1 distant, 2 rect
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

    for (uint i = 0u; i < s.u->object_count; i++) {
        Hit h = isect_object(s.objects[i], o, d);
        if (h.valid && h.t < best.t) {
            best.hit = true;
            best.t = h.t;
            best.p = h.p;
            best.n = h.n;
            best.material = s.object_materials[i];
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

// ---- BSDF lobes (mirror of pt/mod.rs) ----

inline void lobes_of(PtMaterial m, thread float3& albedo, thread float& alpha,
                     thread float& p_spec) {
    albedo = (m.kind == 2u) ? float3(0.0f) : float3(m.color[0], m.color[1], m.color[2]);
    alpha = m.alpha;
    p_spec = m.p_spec;
}

inline float3 f0_of(PtMaterial m) {
    return (m.kind == 2u) ? float3(m.color[0], m.color[1], m.color[2])
                          : float3(0.04f);
}

inline float3 fresnel_schlick(float3 f0, float c) {
    float mfac = pow(clamp(1.0f - c, 0.0f, 1.0f), 5.0f);
    return f0 + (float3(1.0f) - f0) * mfac;
}

inline float ggx_d(float ndh, float alpha) {
    float a2 = alpha * alpha;
    float dd = ndh * ndh * (a2 - 1.0f) + 1.0f;
    return a2 / (PT_PI * dd * dd);
}

inline float ggx_g1(float ndv, float alpha) {
    float a2 = alpha * alpha;
    float denom = ndv + sqrt(a2 + (1.0f - a2) * ndv * ndv);
    return denom <= 0.0f ? 0.0f : 2.0f * ndv / denom;
}

inline void basis_of(float3 n, thread float3& b1, thread float3& b2) {
    float3 t = (fabs(n.x) > 0.9f) ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    b1 = normalize_cpu(cross(n, t));
    b2 = cross(n, b1);
}

inline float3 cosine_sample(float3 n, thread Pcg32& rng) {
    float u = pcg_f32(rng);
    float v = pcg_f32(rng);
    float r = sqrt(u);
    float phi = 2.0f * PT_PI * v;
    float3 b1, b2;
    basis_of(n, b1, b2);
    return normalize_cpu(b1 * (r * cos(phi)) + b2 * (r * sin(phi)) + n * sqrt(1.0f - u));
}

inline float3 ggx_sample_half(float3 n, float alpha, thread Pcg32& rng) {
    float u = pcg_f32(rng);
    float v = pcg_f32(rng);
    float phi = 2.0f * PT_PI * v;
    float ct = sqrt((1.0f - u) / (u * (alpha * alpha - 1.0f) + 1.0f));
    float st = sqrt(max(1.0f - ct * ct, 0.0f));
    float3 b1, b2;
    basis_of(n, b1, b2);
    return normalize_cpu(b1 * (st * cos(phi)) + b2 * (st * sin(phi)) + n * ct);
}

inline float3 bsdf_eval(PtMaterial m, float3 wo, float3 wi, float3 n) {
    float ndo = max(dot(n, wo), 0.0f);
    float ndi = max(dot(n, wi), 0.0f);
    if (ndo <= 0.0f || ndi <= 0.0f) return float3(0.0f);
    float3 albedo;
    float alpha, p_spec;
    lobes_of(m, albedo, alpha, p_spec);
    float3 f = float3(0.0f);
    if (p_spec < 1.0f) f += albedo * (1.0f / PT_PI);
    if (p_spec > 0.0f) {
        float3 h = normalize_cpu(wo + wi);
        float ndh = max(dot(n, h), 0.0f);
        float odh = max(dot(wo, h), 1e-9f);
        float dtr = ggx_d(ndh, alpha);
        float g = ggx_g1(ndo, alpha) * ggx_g1(ndi, alpha);
        float3 fr = fresnel_schlick(f0_of(m), odh);
        f += fr * (dtr * g / max(4.0f * ndo * ndi, 1e-9f));
    }
    return f;
}

inline float bsdf_pdf(PtMaterial m, float3 wo, float3 wi, float3 n) {
    float ndi = max(dot(n, wi), 0.0f);
    if (ndi <= 0.0f) return 0.0f;
    float3 albedo;
    float alpha, p_spec;
    lobes_of(m, albedo, alpha, p_spec);
    float pdf_diffuse = ndi / PT_PI;
    if (p_spec <= 0.0f) return pdf_diffuse;
    float3 h = normalize_cpu(wo + wi);
    float ndh = max(dot(n, h), 0.0f);
    float odh = max(fabs(dot(wo, h)), 1e-9f);
    float pdf_spec = ggx_d(ndh, alpha) * ndh / (4.0f * odh);
    return (1.0f - p_spec) * pdf_diffuse + p_spec * pdf_spec;
}

inline bool sample_bsdf(PtMaterial m, float3 wo, float3 n, thread Pcg32& rng,
                        thread float3& wi_out, thread float3& f_out,
                        thread float& pdf_out) {
    float3 albedo;
    float alpha, p_spec;
    lobes_of(m, albedo, alpha, p_spec);
    float3 wi;
    if (pcg_f32(rng) < p_spec) {
        float3 h = ggx_sample_half(n, alpha, rng);
        wi = reflect(-wo, h);
        if (dot(wi, n) <= 0.0f) return false;
    } else {
        wi = cosine_sample(n, rng);
    }
    float pdf = bsdf_pdf(m, wo, wi, n);
    if (pdf <= 0.0f) return false;
    wi_out = wi;
    f_out = bsdf_eval(m, wo, wi, n);
    pdf_out = pdf;
    return true;
}

// ---- lights ----

inline float power_heuristic(float a, float b) {
    float a2 = a * a;
    float b2 = b * b;
    return (a2 + b2 <= 0.0f) ? 0.0f : a2 / (a2 + b2);
}

inline float rect_pdf_solid_angle(PtLight l, float3 origin, float3 hit_p) {
    float3 d = hit_p - origin;
    float dist2 = dot(d, d);
    float3 ln = float3(l.normal[0], l.normal[1], l.normal[2]);
    float cl = fabs(dot(normalize_cpu(d), ln));
    if (cl < 1e-9f || l.area <= 0.0f) return 0.0f;
    return dist2 / (cl * l.area);
}

inline float3 sample_light(thread const PtScene& s, PtLight l, float3 p,
                           float3 n, float3 wo, PtMaterial m,
                           thread Pcg32& rng) {
    float3 rad = float3(l.radiance[0], l.radiance[1], l.radiance[2]);
    if (l.kind == 0u) {                      // point, inverse-square
        float3 pos = float3(l.a[0], l.a[1], l.a[2]);
        float3 to_l = pos - p;
        float dist2 = max(dot(to_l, to_l), 1e-12f);
        float dist = sqrt(dist2);
        float3 wi = to_l / dist;
        float c = max(dot(wi, n), 0.0f);
        if (c <= 0.0f || pt_occluded(s, p, n, wi, dist)) return float3(0.0f);
        return bsdf_eval(m, wo, wi, n) * rad * (c / dist2);
    }
    if (l.kind == 1u) {                      // distant
        float3 wi = -float3(l.a[0], l.a[1], l.a[2]);
        float c = max(dot(wi, n), 0.0f);
        if (c <= 0.0f || pt_occluded(s, p, n, wi, PT_BIG)) return float3(0.0f);
        return bsdf_eval(m, wo, wi, n) * rad * c;
    }
    // rect with MIS
    float u = pcg_f32(rng);
    float v = pcg_f32(rng);
    float3 corner = float3(l.a[0], l.a[1], l.a[2]);
    float3 e1 = float3(l.e1[0], l.e1[1], l.e1[2]);
    float3 e2 = float3(l.e2[0], l.e2[1], l.e2[2]);
    float3 sp = corner + e1 * u + e2 * v;
    float3 to_l = sp - p;
    float dist2 = max(dot(to_l, to_l), 1e-12f);
    float dist = sqrt(dist2);
    float3 wi = to_l / dist;
    float cs = max(dot(wi, n), 0.0f);
    float3 ln = float3(l.normal[0], l.normal[1], l.normal[2]);
    float cl = fabs(dot(wi, ln));
    if (cs <= 0.0f || cl < 1e-9f || l.area <= 0.0f) return float3(0.0f);
    if (pt_occluded(s, p, n, wi, dist)) return float3(0.0f);
    float pdf_sa = dist2 / (cl * l.area);
    float3 f = bsdf_eval(m, wo, wi, n);
    float bp = bsdf_pdf(m, wo, wi, n);
    float w = power_heuristic(pdf_sa, bp);
    return f * rad * (cs / pdf_sa) * w;
}

// ---- the path integrator (mirror of pt::trace) ----

inline float3 pt_li(thread const PtScene& s, float3 origin, float3 dir,
                    thread Pcg32& rng) {
    constant PtUniforms& u = *s.u;
    float3 l = float3(0.0f);
    float3 beta = float3(1.0f);
    float prev_pdf = 0.0f;
    float3 prev_origin = origin;
    bool from_camera = true;

    for (uint depth = 0u; depth < u.max_bounces; depth++) {
        PtHit h = pt_trace_scene(s, origin, dir);
        if (!h.hit) {
            l += beta * float3(u.background[0], u.background[1], u.background[2]);
            break;
        }
        PtMaterial m = s.materials[h.material];
        float3 wo = -normalize_cpu(dir);
        float3 n = (dot(h.n, wo) < 0.0f) ? -h.n : h.n;

        float3 em = float3(m.emission[0], m.emission[1], m.emission[2]);
        if (max(max(em.x, em.y), em.z) > 0.0f) {
            float w = 1.0f;
            if (!from_camera && m.area_light != 0xFFFFFFFFu) {
                float pdf_light = rect_pdf_solid_angle(s.lights[m.area_light],
                                                       prev_origin, h.p)
                    / (float)u.light_count;
                w = power_heuristic(prev_pdf, pdf_light);
            }
            l += beta * em * w;
        }

        if (u.light_count > 0u) {
            uint li = min((uint)(pcg_f32(rng) * (float)u.light_count),
                          u.light_count - 1u);
            float3 c = sample_light(s, s.lights[li], h.p, n, wo, m, rng);
            l += beta * c * (float)u.light_count;
        }

        float3 wi, f;
        float pdf;
        if (!sample_bsdf(m, wo, n, rng, wi, f, pdf)) break;
        float c = max(dot(wi, n), 0.0f);
        if (pdf <= 0.0f || c <= 0.0f) break;
        beta *= f * (c / pdf);
        prev_pdf = pdf;
        prev_origin = h.p;
        from_camera = false;

        origin = h.p + n * PT_RAY_OFFSET;
        dir = wi;

        if (depth >= u.rr_start) {
            float q = min(max(max(beta.x, beta.y), beta.z), 0.95f);
            if (q <= 0.0f || pcg_f32(rng) > q) break;
            beta /= q;
        }
    }
    return l;
}

// ---- entry: one thread per pixel, a batch of samples per dispatch ----

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
                      constant PtUniforms&     u                [[buffer(11)]],
                      device float*            accum            [[buffer(12)]],
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
    s.u = &u;

    float3 eye = float3(u.eye[0], u.eye[1], u.eye[2]);
    float3 fwd = float3(u.forward[0], u.forward[1], u.forward[2]);
    float3 rgt = float3(u.right[0], u.right[1], u.right[2]);
    float3 upv = float3(u.up[0], u.up[1], u.up[2]);

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
        float3 dir = normalize_cpu(fwd + rgt * (uu * u.half_width) + upv * (vv * u.half_height));
        float3 li = pt_li(s, eye, dir, rng);
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
