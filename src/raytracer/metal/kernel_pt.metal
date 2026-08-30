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
    float lens_radius;      // 0 = pinhole
    float focal_distance;
    uint  projection;       // 0 perspective, 1 orthographic
    float ortho_half_w;
    float ortho_half_h;
    uint  filter_kind;      // 0 box, 1 triangle, 2 gaussian
    float filter_width;
    uint  has_motion;
    uint  light_bvh_count;  // 0 = no finite-light BVH
    float p_infinite;
    float infinite_total;
    uint  pad_ls;
    uint  atmosphere;       // global medium index, 0xFFFFFFFF = none
    uint  media_count;
    uint  pad_med[2];
};

// PtMaterial is defined in pattern_prelude.metal (compiled just before
// this file) because the generated pattern code mutates it.

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
    uint has_st;
    uint has_deform;    // vertices1 holds shutter-close positions
};

struct InstanceG {
    float inv[16];
    float fwd[16];
    float fwd1[16];      // transform-motion endpoint (== fwd when static)
    uint  mesh_id;
    uint  material_id;
    float scale;         // isotropic transform scale (st-density transfer)
    uint  has_motion;
    uint  kind;          // 0 mesh, 1 curve set
    uint  pad[3];
};

struct CurveSegG {
    float p0[4];         // xyz + radius
    float p1[4];
    float v0;
    float v1;
    float pad[2];
};

struct CurveInfoG {
    uint node_offset;
    uint seg_offset;
    uint pad[2];
};

struct LightBvhNodeG {
    float mn[3];
    float power;
    float mx[3];
    uint parent;
    uint a;          // interior: left child; leaf: light index
    uint b;          // 0xFFFFFFFF marks a leaf
    uint pad[2];
};

struct LightAuxG {
    uint leaf;       // light BVH leaf, 0xFFFFFFFF for infinite lights
    float inf_weight;
};

struct MediumG {
    float sigma_a[3];
    float g;
    float sigma_s[3];
    float majorant;
    float emission[3];
    uint  has_density;
    float frequency;
    uint  octaves;
    float gain;
    float lacunarity;
    float coverage;
    float sharpness;
    float max_distance;
    float pad;
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
    device const float*     st;
    device const float*     vertices1;
    device const CurveSegG* curve_segs;
    device const CurveInfoG* curve_infos;
    device const LightBvhNodeG* light_bvh;
    device const LightAuxG* light_aux;
    device const MediumG* media;
    device const float*     tex_data;
    device const TexMipG*   tex_mips;
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

// ---- participating media (mirror of pt/volume.rs + scene/medium.rs) -----

inline float med_density(device const MediumG& m, float3 p) {
    if (m.has_density == 0u) return 1.0f;
    float n = pat_fbm(p, m.frequency, m.octaves, m.gain, m.lacunarity) * 0.5f + 0.5f;
    return clamp((n - (1.0f - m.coverage)) * m.sharpness, 0.0f, 1.0f);
}

inline float hg_phase_g(float g, float cos_theta) {
    float denom = 1.0f + g * g + 2.0f * g * cos_theta;
    return (1.0f - g * g) / (4.0f * PT_PI * denom * sqrt(max(denom, 1e-9f)));
}

inline float3 hg_sample_g(float g, float3 wo, float u1, float u2,
                          thread float& pdf_out) {
    float cos_theta;
    if (fabs(g) < 1e-3f) {
        cos_theta = 1.0f - 2.0f * u1;
    } else {
        float sq = (1.0f - g * g) / (1.0f + g - 2.0f * g * u1);
        cos_theta = -(1.0f + g * g - sq * sq) / (2.0f * g);
    }
    float sin_theta = sqrt(max(1.0f - cos_theta * cos_theta, 0.0f));
    float phi = 2.0f * PT_PI * u2;
    float3 w = -wo;
    float3 a = fabs(w.x) > 0.9f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    float3 t = normalize_cpu(cross(w, a));
    float3 b = cross(w, t);
    float3 wi = normalize_cpu(t * (sin_theta * cos(phi)) + b * (sin_theta * sin(phi))
        + w * cos_theta);
    pdf_out = hg_phase_g(g, -dot(wo, wi));
    return wi;
}

// Distance sampling. Returns true on scatter (t_out, weight_out set);
// false on pass (weight_out set).
inline bool med_sample_distance(device const MediumG& m, float3 o, float3 d,
                                float t_max, float3 beta, thread Pcg32& rng,
                                thread float& t_out, thread float3& weight_out) {
    t_max = min(t_max, m.max_distance);
    float3 st = float3(m.sigma_a[0] + m.sigma_s[0],
                       m.sigma_a[1] + m.sigma_s[1],
                       m.sigma_a[2] + m.sigma_s[2]);
    float3 ss = float3(m.sigma_s[0], m.sigma_s[1], m.sigma_s[2]);
    if (m.has_density == 0u) {
        float st_avg = (st.x + st.y + st.z) / 3.0f;
        if (st_avg <= 1e-9f) { weight_out = float3(1.0f); return false; }
        float3 b = max(beta, float3(0.0f));
        float total = b.x + b.y + b.z;
        float3 w = total > 1e-12f ? b / total : float3(1.0f / 3.0f);
        float u = pcg_f32(rng);
        float sigma_c = u < w.x ? st.x : (u < w.x + w.y ? st.y : st.z);
        sigma_c = max(sigma_c, 1e-9f);
        float t = -log(max(1.0f - pcg_f32(rng), 1e-12f)) / sigma_c;
        if (t < t_max) {
            float3 tr = exp(-st * t);
            float pdf = w.x * st.x * tr.x + w.y * st.y * tr.y + w.z * st.z * tr.z;
            if (pdf <= 1e-30f) { weight_out = float3(0.0f); return false; }
            t_out = t;
            weight_out = ss * tr / pdf;
            return true;
        }
        float3 tr = exp(-st * t_max);
        float pdf = w.x * tr.x + w.y * tr.y + w.z * tr.z;
        weight_out = pdf > 1e-30f ? tr / pdf : float3(0.0f);
        return false;
    }
    // Weighted delta tracking.
    float majorant = m.majorant;
    if (majorant <= 1e-9f) { weight_out = float3(1.0f); return false; }
    float t = 0.0f;
    float3 weight = float3(1.0f);
    for (int i = 0; i < 4096; i++) {
        t -= log(max(1.0f - pcg_f32(rng), 1e-12f)) / majorant;
        if (t >= t_max) { weight_out = weight; return false; }
        float rho = med_density(m, o + d * t);
        float3 st_here = st * rho;
        float p_real = clamp((st_here.x + st_here.y + st_here.z) / (3.0f * majorant),
                             0.0f, 1.0f);
        if (pcg_f32(rng) < p_real) {
            t_out = t;
            weight_out = weight * (ss * rho) / (majorant * p_real);
            return true;
        }
        float p_null = max(1.0f - p_real, 1e-9f);
        weight *= (float3(1.0f) - st_here / majorant) / p_null;
        if (max(max(weight.x, weight.y), weight.z) < 1e-6f) {
            weight_out = float3(0.0f);
            return false;
        }
    }
    weight_out = weight;
    return false;
}

// Ratio-tracked (or closed-form) transmittance of one medium segment.
inline float3 med_transmittance(device const MediumG& m, float3 o, float3 d,
                                float dist, thread Pcg32& rng) {
    dist = min(dist, m.max_distance);
    float3 st = float3(m.sigma_a[0] + m.sigma_s[0],
                       m.sigma_a[1] + m.sigma_s[1],
                       m.sigma_a[2] + m.sigma_s[2]);
    if (m.has_density == 0u) return exp(-st * dist);
    float majorant = m.majorant;
    if (majorant <= 1e-9f) return float3(1.0f);
    float t = 0.0f;
    float3 tr = float3(1.0f);
    for (int i = 0; i < 4096; i++) {
        t -= log(max(1.0f - pcg_f32(rng), 1e-12f)) / majorant;
        if (t >= dist) break;
        float rho = med_density(m, o + d * t);
        tr *= float3(1.0f) - st * (rho / majorant);
        if (max(max(tr.x, tr.y), tr.z) < 1e-5f) return float3(0.0f);
    }
    return tr;
}

// A volume hull: interior medium, no lobes, no emission — pure boundary.
inline bool is_volume_hull(thread const PtMaterial& m) {
    float wsum = m.weights[0] + m.weights[1] + m.weights[2] + m.weights[3]
        + m.weights[4];
    float em = m.emission[0] + m.emission[1] + m.emission[2];
    return m.interior != 0xFFFFFFFFu && m.is_hair == 0u && wsum <= 0.0f && em <= 0.0f;
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
    float2 st;           // surface parameterization (meshes; quadrics 0)
    float st_density;    // st units per world unit (0 = no st)
    float3 tangent;      // fiber tangent (curves; zero for surfaces)
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

inline Affine affine_lerp(Affine a, Affine b, float t) {
    Affine o;
    o.r0 = mix(a.r0, b.r0, t);
    o.r1 = mix(a.r1, b.r1, t);
    o.r2 = mix(a.r2, b.r2, t);
    return o;
}

// Inverse of an affine transform: adjugate of the 3x3 + back-solved
// translation (mirrors Matrix4::inverse for the affine case).
inline Affine affine_inverse(Affine a) {
    float3 c0 = float3(a.r0.x, a.r1.x, a.r2.x);
    float3 c1 = float3(a.r0.y, a.r1.y, a.r2.y);
    float3 c2 = float3(a.r0.z, a.r1.z, a.r2.z);
    float3 tr = float3(a.r0.w, a.r1.w, a.r2.w);
    float3 r0 = cross(c1, c2);
    float3 r1 = cross(c2, c0);
    float3 r2 = cross(c0, c1);
    float det = dot(c0, r0);
    float inv_det = 1.0f / (fabs(det) > 1e-20f ? det : 1e-20f);
    r0 *= inv_det; r1 *= inv_det; r2 *= inv_det;
    Affine o;
    o.r0 = float4(r0, -dot(r0, tr));
    o.r1 = float4(r1, -dot(r1, tr));
    o.r2 = float4(r2, -dot(r2, tr));
    return o;
}

// Instance inverse at shutter time (lerp + invert only when moving).
inline Affine instance_inverse_at(device const InstanceG& inst, float time) {
    if (inst.has_motion != 0u && time > 0.0f) {
        return affine_inverse(
            affine_lerp(load_affine(inst.fwd), load_affine(inst.fwd1), time));
    }
    return load_affine(inst.inv);
}

// Mesh vertex at shutter time (deformation lerp).
inline float3 fetch_vertex_at(thread const PtScene& s, device const MeshInfoG& mesh,
                              uint idx, float time) {
    float3 v0 = fetch_v3(s.vertices, idx);
    if (mesh.has_deform != 0u && time > 0.0f) {
        float3 v1 = fetch_v3(s.vertices1, idx);
        return mix(v0, v1, time);
    }
    return v0;
}

// Ray vs rounded cone (sphere-swept segment, lerped radius), mirroring
// curves.rs. Direction unnormalized: t is parametric.
inline bool capsule_hit(float3 o, float3 d, float3 pa, float ra, float3 pb, float rb,
                        float t_max, thread float& t_out, thread float3& n_out) {
    float3 ba = pb - pa;
    float3 oa = o - pa;
    float rr = ra - rb;
    float m0 = dot(ba, ba);
    float m1 = dot(ba, oa);
    float m2 = dot(ba, d);
    float m3 = dot(d, oa);
    float m5 = dot(oa, oa);
    float dd = dot(d, d);

    bool found = false;
    if (m0 < 1e-18f) {
        // Degenerate: sphere of radius max(ra, rb).
        float r = max(ra, rb);
        float b = m3;
        float c = m5 - r * r;
        float disc = b * b - dd * c;
        if (disc < 0.0f) return false;
        float sq = sqrt(disc);
        float t = (-b - sq) / dd;
        if (t <= 1e-6f) t = (-b + sq) / dd;
        if (t <= 1e-6f || t >= t_max) return false;
        t_out = t;
        n_out = normalize_cpu(o + d * t - pa);
        return true;
    }

    float d2 = m0 - rr * rr;
    float k2 = d2 * dd - m2 * m2;
    float k1 = d2 * m3 - m1 * m2 + m2 * rr * ra;
    float k0 = d2 * m5 - m1 * m1 + m1 * rr * ra * 2.0f - m0 * ra * ra;
    if (fabs(k2) > 1e-18f) {
        float h = k1 * k1 - k0 * k2;
        if (h >= 0.0f) {
            float t = (-k1 - sqrt(h)) / k2;
            float y = m1 + t * m2;
            if (t > 1e-6f && t < t_max && y > 0.0f && y < d2) {
                t_out = t;
                n_out = normalize_cpu((oa + d * t) * d2 - ba * y);
                return true;
            }
        }
    }
    // Caps: start owns y < 0, end owns y > d2.
    float t_lim = t_max;
    {
        float b = m3;
        float c = m5 - ra * ra;
        float disc = b * b - dd * c;
        if (disc >= 0.0f) {
            float sq = sqrt(disc);
            float t = (-b - sq) / dd;
            if (t <= 1e-6f) t = (-b + sq) / dd;
            if (t > 1e-6f && t < t_lim && (m1 + t * m2) <= 0.0f) {
                t_out = t;
                n_out = normalize_cpu(o + d * t - pa);
                t_lim = t;
                found = true;
            }
        }
    }
    {
        float3 ob = o - pb;
        float b = dot(d, ob);
        float c = dot(ob, ob) - rb * rb;
        float disc = b * b - dd * c;
        if (disc >= 0.0f) {
            float sq = sqrt(disc);
            float t = (-b - sq) / dd;
            if (t <= 1e-6f) t = (-b + sq) / dd;
            if (t > 1e-6f && t < t_lim && (m1 + t * m2) >= d2) {
                t_out = t;
                n_out = normalize_cpu(o + d * t - pb);
                found = true;
            }
        }
    }
    return found;
}

// Closest hit within one curve set's BLAS (mirrors instance_hit).
inline bool curve_instance_hit(thread const PtScene& s, uint inst_id,
                               float3 wo_pos, float3 wd, float time, float t_max,
                               thread float& t_out, thread float3& n_out,
                               thread float3& tangent_out, thread float& v_out) {
    device const InstanceG& inst = s.instances[inst_id];
    device const CurveInfoG& info = s.curve_infos[inst.mesh_id];
    Affine inv = instance_inverse_at(inst, time);
    float3 o = xf_point(inv, wo_pos);
    float3 d = xf_vec(inv, wd);
    float3 inv_d = safe_inv(d);

    float best_t = t_max;
    uint best_seg = 0xFFFFFFFFu;
    float3 best_n = float3(0.0f);

    uint stack[48];
    int sp = 0;
    stack[sp++] = 0u;
    while (sp > 0) {
        device const BvhNodeG& node = s.blas[info.node_offset + stack[--sp]];
        float tn;
        if (!aabb_hit(node, o, inv_d, best_t, tn)) continue;
        if (node.count > 0u) {
            for (uint i = 0u; i < node.count; i++) {
                uint seg = info.seg_offset + node.left_or_first + i;
                device const CurveSegG& cs = s.curve_segs[seg];
                float t;
                float3 n;
                if (capsule_hit(o, d,
                                float3(cs.p0[0], cs.p0[1], cs.p0[2]), cs.p0[3],
                                float3(cs.p1[0], cs.p1[1], cs.p1[2]), cs.p1[3],
                                best_t, t, n)) {
                    best_t = t;
                    best_seg = seg;
                    best_n = n;
                }
            }
        } else {
            uint l = node.left_or_first;
            uint r = l + 1u;
            float tl, tr;
            bool hl = aabb_hit(s.blas[info.node_offset + l], o, inv_d, best_t, tl);
            bool hr = aabb_hit(s.blas[info.node_offset + r], o, inv_d, best_t, tr);
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
    if (best_seg == 0xFFFFFFFFu) return false;
    device const CurveSegG& cs = s.curve_segs[best_seg];
    float3 pa = float3(cs.p0[0], cs.p0[1], cs.p0[2]);
    float3 pb = float3(cs.p1[0], cs.p1[1], cs.p1[2]);
    float3 axis = pb - pa;
    float len2 = dot(axis, axis);
    float3 hitp = o + d * best_t;
    float along = len2 > 1e-18f ? clamp(dot(hitp - pa, axis) / len2, 0.0f, 1.0f) : 0.5f;
    v_out = cs.v0 + (cs.v1 - cs.v0) * along;
    float3 tang = len2 > 1e-18f ? normalize_cpu(axis) : float3(0.0f, 0.0f, 1.0f);
    t_out = best_t;
    n_out = normalize_cpu(xf_normal(inv, best_n));
    // Tangent transforms with the forward matrix.
    Affine fwd_a = load_affine(inst.fwd);
    tangent_out = normalize_cpu(xf_vec(fwd_a, tang));
    return true;
}

// Any curve hit before t_limit (shadow rays).
inline bool curve_instance_occludes(thread const PtScene& s, uint inst_id,
                                    float3 wo_pos, float3 wd, float time, float t_limit) {
    device const InstanceG& inst = s.instances[inst_id];
    device const CurveInfoG& info = s.curve_infos[inst.mesh_id];
    Affine inv = instance_inverse_at(inst, time);
    float3 o = xf_point(inv, wo_pos);
    float3 d = xf_vec(inv, wd);
    float3 inv_d = safe_inv(d);
    uint stack[48];
    int sp = 0;
    stack[sp++] = 0u;
    while (sp > 0) {
        device const BvhNodeG& node = s.blas[info.node_offset + stack[--sp]];
        float tn;
        if (!aabb_hit(node, o, inv_d, t_limit, tn)) continue;
        if (node.count > 0u) {
            for (uint i = 0u; i < node.count; i++) {
                uint seg = info.seg_offset + node.left_or_first + i;
                device const CurveSegG& cs = s.curve_segs[seg];
                float t;
                float3 n;
                if (capsule_hit(o, d,
                                float3(cs.p0[0], cs.p0[1], cs.p0[2]), cs.p0[3],
                                float3(cs.p1[0], cs.p1[1], cs.p1[2]), cs.p1[3],
                                t_limit, t, n)) return true;
            }
        } else {
            stack[sp++] = node.left_or_first;
            stack[sp++] = node.left_or_first + 1u;
        }
    }
    return false;
}

// Closest hit within one instance's BLAS. t stays parametric along the
// world direction (the instance transform is applied without normalizing).
inline bool instance_hit(thread const PtScene& s, uint inst_id,
                         float3 wo_pos, float3 wd, float time, float t_max,
                         thread float& t_out, thread float3& n_out,
                         thread float2& st_out, thread float& st_density_out) {
    device const InstanceG& inst = s.instances[inst_id];
    device const MeshInfoG& mesh = s.mesh_infos[inst.mesh_id];
    Affine inv = instance_inverse_at(inst, time);
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
                float3 v0 = fetch_vertex_at(s, mesh, mesh.vertex_offset + s.tri_indices[base], time);
                float3 v1 = fetch_vertex_at(s, mesh, mesh.vertex_offset + s.tri_indices[base + 1u], time);
                float3 v2 = fetch_vertex_at(s, mesh, mesh.vertex_offset + s.tri_indices[base + 2u], time);
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
    // Deforming meshes use the time-correct geometric normal (authored
    // normals describe the rest pose) — mirrors Mesh::local_normal.
    if (mesh.has_normals != 0u && mesh.has_deform == 0u) {
        float3 n0 = fetch_v3(s.normals, mesh.vertex_offset + i0);
        float3 n1 = fetch_v3(s.normals, mesh.vertex_offset + i1);
        float3 n2 = fetch_v3(s.normals, mesh.vertex_offset + i2);
        float w = 1.0f - best_u - best_v;
        nl = normalize_cpu(n0 * w + n1 * best_u + n2 * best_v);
    } else {
        float3 v0 = fetch_vertex_at(s, mesh, mesh.vertex_offset + i0, time);
        float3 v1 = fetch_vertex_at(s, mesh, mesh.vertex_offset + i1, time);
        float3 v2 = fetch_vertex_at(s, mesh, mesh.vertex_offset + i2, time);
        nl = normalize_cpu(cross(v1 - v0, v2 - v0));
    }
    t_out = best_t;
    n_out = normalize_cpu(xf_normal(inv, nl));

    // st: interpolated coordinates + density (st area over geometric
    // area, scaled by the instance's isotropic scale) — mirrors
    // Mesh::st_at + Instance::intersect on the CPU.
    st_out = float2(0.0f);
    st_density_out = 0.0f;
    if (mesh.has_st != 0u) {
        float2 s0 = float2(s.st[(mesh.vertex_offset + i0) * 2u],
                           s.st[(mesh.vertex_offset + i0) * 2u + 1u]);
        float2 s1 = float2(s.st[(mesh.vertex_offset + i1) * 2u],
                           s.st[(mesh.vertex_offset + i1) * 2u + 1u]);
        float2 s2 = float2(s.st[(mesh.vertex_offset + i2) * 2u],
                           s.st[(mesh.vertex_offset + i2) * 2u + 1u]);
        float w = 1.0f - best_u - best_v;
        st_out = s0 * w + s1 * best_u + s2 * best_v;
        float st_area = 0.5f * fabs((s1.x - s0.x) * (s2.y - s0.y)
                                  - (s2.x - s0.x) * (s1.y - s0.y));
        float3 v0 = fetch_v3(s.vertices, mesh.vertex_offset + i0);
        float3 v1 = fetch_v3(s.vertices, mesh.vertex_offset + i1);
        float3 v2 = fetch_v3(s.vertices, mesh.vertex_offset + i2);
        float geo_area = 0.5f * length(cross(v1 - v0, v2 - v0));
        if (geo_area > 1e-18f) {
            st_density_out = sqrt(st_area / geo_area) / max(inst.scale, 1e-12f);
        }
    }
    return true;
}

// Any triangle hit within one instance before t_limit (shadow rays).
inline bool instance_occludes(thread const PtScene& s, uint inst_id,
                              float3 wo_pos, float3 wd, float time, float t_limit) {
    device const InstanceG& inst = s.instances[inst_id];
    device const MeshInfoG& mesh = s.mesh_infos[inst.mesh_id];
    Affine inv = instance_inverse_at(inst, time);
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
                float3 v0 = fetch_vertex_at(s, mesh, mesh.vertex_offset + s.tri_indices[base], time);
                float3 v1 = fetch_vertex_at(s, mesh, mesh.vertex_offset + s.tri_indices[base + 1u], time);
                float3 v2 = fetch_vertex_at(s, mesh, mesh.vertex_offset + s.tri_indices[base + 2u], time);
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
inline PtHit pt_trace_scene(thread const PtScene& s, float3 o, float3 d, float time) {
    PtHit best;
    best.hit = false;
    best.t = PT_BIG;
    best.p = float3(0.0f);
    best.n = float3(0.0f);
    best.material = 0u;
    best.front = true;
    best.st = float2(0.0f);
    best.st_density = 0.0f;
    best.tangent = float3(0.0f);

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
                    if (s.instances[inst_id].kind == 1u) {
                        float t;
                        float3 n;
                        float3 tang;
                        float v;
                        if (curve_instance_hit(s, inst_id, o, d, time, best.t, t, n, tang, v)) {
                            best.hit = true;
                            best.t = t;
                            best.p = o + d * t;
                            best.front = dot(n, d) < 0.0f;
                            best.n = n;
                            best.material = s.instances[inst_id].material_id;
                            best.st = float2(0.5f, v);
                            best.st_density = 0.0f;
                            best.tangent = tang;
                        }
                        continue;
                    }
                    float t;
                    float3 n;
                    float2 st;
                    float st_density;
                    if (instance_hit(s, inst_id, o, d, time, best.t, t, n, st, st_density)) {
                        best.hit = true;
                        best.t = t;
                        best.p = o + d * t;
                        // Unflipped mesh normal: side falls out of the dot.
                        best.front = dot(n, d) < 0.0f;
                        best.n = n;
                        best.material = s.instances[inst_id].material_id;
                        best.st = st;
                        best.st_density = st_density;
                        best.tangent = float3(0.0f);
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
                        float3 dir, float dist, float time) {
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
                    uint inst_id = node.left_or_first + i;
                    bool blocked = s.instances[inst_id].kind == 1u
                        ? curve_instance_occludes(s, inst_id, o, dir, time, limit)
                        : instance_occludes(s, inst_id, o, dir, time, limit);
                    if (blocked) return true;
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
                        thread float& pdf_out, thread bool& transmitted,
                        thread bool& spec_lobe) {
    float wd = m.weights[0], ws = m.weights[1], wc = m.weights[2],
          wf = m.weights[3], wg = m.weights[4];
    if (wd + ws + wc + wf + wg <= 0.0f) return false;
    transmitted = false;
    float pick = pcg_f32(rng);
    spec_lobe = pick >= wd + wf;
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

// Fractional visibility: presence cutouts attenuate, opaque kills,
// volume hulls toggle the medium, media tint the segments.
inline float3 pt_visibility(thread const PtScene& s, float3 p, float3 n,
                            float3 dir, float dist, float time, uint medium,
                            thread Pcg32& rng) {
    float3 origin = p + n * PT_RAY_OFFSET;
    float remaining = dist - 1e-3f;
    float3 vis = float3(1.0f);
    for (int i = 0; i < 16; i++) {
        PtHit h = pt_trace_scene(s, origin, dir, time);
        float seg = h.hit ? min(h.t, remaining) : remaining;
        if (medium != 0xFFFFFFFFu) {
            vis *= med_transmittance(s.media[medium], origin, dir, seg, rng);
            if (max(max(vis.x, vis.y), vis.z) < 1e-4f) return float3(0.0f);
        }
        if (!h.hit || h.t >= remaining) return vis;
        PtMaterial pm = s.materials[h.material];
        if (is_volume_hull(pm)) {
            medium = h.front ? pm.interior : s.u->atmosphere;
            origin = h.p + dir * PT_RAY_OFFSET;
            remaining -= h.t + PT_RAY_OFFSET;
            continue;
        }
        apply_patterns(h.material, pm, h.st, h.p, h.n, 0.0f, s.tex_data, s.tex_mips);
        float presence = clamp(pm.presence, 0.0f, 1.0f);
        if (presence >= 1.0f) return float3(0.0f);
        vis *= 1.0f - presence;
        if (max(max(vis.x, vis.y), vis.z) < 1e-4f) return float3(0.0f);
        origin = h.p + dir * PT_RAY_OFFSET;
        remaining -= h.t + PT_RAY_OFFSET;
    }
    return vis;
}


// ---- Marschner/d'Eon hair scattering (mirror of pt/hair.rs) -------------
// Fiber frame: x = tangent, yz = normal plane. f is per-solid-angle with
// no cosine factors on either side.

constant int HAIR_PMAX = 3;

inline float hair_i0(float x) {
    float val = 0.0f;
    float x2i = 1.0f;
    float ifact = 1.0f;
    float i4 = 1.0f;
    for (int i = 0; i < 10; i++) {
        if (i > 1) ifact *= (float)i;
        val += x2i / (i4 * ifact * ifact);
        x2i *= x * x;
        i4 *= 4.0f;
    }
    return val;
}

inline float hair_log_i0(float x) {
    if (x > 12.0f) return x + 0.5f * (-log(2.0f * PT_PI * x) + 1.0f / (8.0f * x));
    return log(hair_i0(x));
}

inline float hair_mp(float cti, float cto, float sti, float sto, float v) {
    float a = cti * cto / v;
    float b = sti * sto / v;
    if (v <= 0.1f) {
        return exp(hair_log_i0(a) - b - 1.0f / v + 0.6931f + log(1.0f / (2.0f * v)));
    }
    return (exp(-b) * hair_i0(a)) / (sinh(1.0f / v) * 2.0f * v);
}

inline float hair_fr_dielectric(float ci, float eta) {
    ci = clamp(ci, -1.0f, 1.0f);
    float ei = 1.0f;
    float et = eta;
    if (ci <= 0.0f) { ei = eta; et = 1.0f; ci = -ci; }
    float si = sqrt(max(1.0f - ci * ci, 0.0f));
    float st = ei / et * si;
    if (st >= 1.0f) return 1.0f;
    float ct = sqrt(max(1.0f - st * st, 0.0f));
    float rp = (et * ci - ei * ct) / (et * ci + ei * ct);
    float rs = (ei * ci - et * ct) / (ei * ci + et * ct);
    return (rp * rp + rs * rs) * 0.5f;
}

struct HairGeomG {
    float sin_to, cos_to, phi_o, gamma_o, gamma_t;
    float3 tbody;   // body transmittance
};

inline HairGeomG hair_geom(thread const PtMaterial& m, float3 wo, float hh) {
    HairGeomG g;
    g.sin_to = clamp(wo.x, -1.0f, 1.0f);
    g.cos_to = sqrt(max(1.0f - g.sin_to * g.sin_to, 0.0f));
    g.phi_o = atan2(wo.z, wo.y);
    g.gamma_o = asin(clamp(hh, -1.0f, 1.0f));
    float sin_tt = g.sin_to / m.hair_eta;
    float cos_tt = sqrt(max(1.0f - sin_tt * sin_tt, 0.0f));
    float etap = sqrt(max(m.hair_eta * m.hair_eta - g.sin_to * g.sin_to, 0.0f))
        / max(g.cos_to, 1e-6f);
    float sgt = clamp(hh / etap, -1.0f, 1.0f);
    float cgt = sqrt(max(1.0f - sgt * sgt, 0.0f));
    g.gamma_t = asin(sgt);
    float l = 2.0f * cgt / max(cos_tt, 1e-6f);
    g.tbody = float3(exp(-m.hair_sigma_a[0] * l),
                     exp(-m.hair_sigma_a[1] * l),
                     exp(-m.hair_sigma_a[2] * l));
    return g;
}

inline void hair_ap(float cos_to, float eta, float hh, float3 t,
                    thread float3 ap_out[4]) {
    float cgo = sqrt(max(1.0f - hh * hh, 0.0f));
    float fr = hair_fr_dielectric(cos_to * cgo, eta);
    ap_out[0] = float3(fr);
    ap_out[1] = t * ((1.0f - fr) * (1.0f - fr));
    ap_out[2] = ap_out[1] * t * fr;
    ap_out[3] = ap_out[2] * fr * t
        / max(float3(1.0f) - t * fr, float3(1e-6f));
}

inline float hair_logistic(float x, float sc) {
    x = fabs(x);
    float e = exp(-x / sc);
    return e / (sc * (1.0f + e) * (1.0f + e));
}

inline float hair_logistic_cdf(float x, float sc) {
    return 1.0f / (1.0f + exp(-x / sc));
}

inline float hair_trimmed_logistic(float x, float sc) {
    return hair_logistic(x, sc)
        / (hair_logistic_cdf(PT_PI, sc) - hair_logistic_cdf(-PT_PI, sc));
}

inline float hair_sample_trimmed_logistic(float u, float sc) {
    float k = hair_logistic_cdf(PT_PI, sc) - hair_logistic_cdf(-PT_PI, sc);
    float inner = clamp(u * k + hair_logistic_cdf(-PT_PI, sc), 1e-9f, 1.0f - 1e-9f);
    float x = -sc * log(1.0f / inner - 1.0f);
    return clamp(x, -PT_PI, PT_PI);
}

inline float hair_phi_fn(int pp, float gamma_o, float gamma_t) {
    return 2.0f * (float)pp * gamma_t - 2.0f * gamma_o + (float)pp * PT_PI;
}

inline float hair_np(float phi, int pp, float sc, float gamma_o, float gamma_t) {
    float dphi = phi - hair_phi_fn(pp, gamma_o, gamma_t);
    while (dphi > PT_PI) dphi -= 2.0f * PT_PI;
    while (dphi < -PT_PI) dphi += 2.0f * PT_PI;
    return hair_trimmed_logistic(dphi, sc);
}

inline float3 hair_f(thread const PtMaterial& m, float3 wo, float3 wi, float hh) {
    HairGeomG g = hair_geom(m, wo, hh);
    float sti = clamp(wi.x, -1.0f, 1.0f);
    float cti = sqrt(max(1.0f - sti * sti, 0.0f));
    float phi = atan2(wi.z, wi.y) - g.phi_o;
    float3 ap[4];
    hair_ap(g.cos_to, m.hair_eta, hh, g.tbody, ap);
    float3 outv = float3(0.0f);
    for (int pp = 0; pp <= HAIR_PMAX; pp++) {
        float mpv = hair_mp(cti, g.cos_to, sti, g.sin_to, m.hair_v[pp]);
        float npv = pp < HAIR_PMAX ? hair_np(phi, pp, m.hair_s, g.gamma_o, g.gamma_t)
                                   : 1.0f / (2.0f * PT_PI);
        outv += ap[pp] * (mpv * npv);
    }
    return outv;
}

inline void hair_ap_pdf(thread const PtMaterial& m, thread const HairGeomG& g,
                        float hh, thread float w_out[4]) {
    float3 ap[4];
    hair_ap(g.cos_to, m.hair_eta, hh, g.tbody, ap);
    float total = 0.0f;
    for (int pp = 0; pp <= HAIR_PMAX; pp++) {
        w_out[pp] = 0.2126f * ap[pp].x + 0.7152f * ap[pp].y + 0.0722f * ap[pp].z;
        total += w_out[pp];
    }
    if (total > 1e-12f) {
        for (int pp = 0; pp <= HAIR_PMAX; pp++) w_out[pp] /= total;
    } else {
        w_out[0] = 1.0f;
    }
}

inline float hair_pdf(thread const PtMaterial& m, float3 wo, float3 wi, float hh) {
    HairGeomG g = hair_geom(m, wo, hh);
    float sti = clamp(wi.x, -1.0f, 1.0f);
    float cti = sqrt(max(1.0f - sti * sti, 0.0f));
    float phi = atan2(wi.z, wi.y) - g.phi_o;
    float w[4];
    hair_ap_pdf(m, g, hh, w);
    float outp = 0.0f;
    for (int pp = 0; pp <= HAIR_PMAX; pp++) {
        float mpv = hair_mp(cti, g.cos_to, sti, g.sin_to, m.hair_v[pp]);
        float npv = pp < HAIR_PMAX ? hair_np(phi, pp, m.hair_s, g.gamma_o, g.gamma_t)
                                   : 1.0f / (2.0f * PT_PI);
        outp += w[pp] * mpv * npv;
    }
    return max(outp, 0.0f);
}

inline bool hair_sample(thread const PtMaterial& m, float3 wo, float hh,
                        thread Pcg32& rng, thread float3& wi_out,
                        thread float3& f_out, thread float& pdf_out) {
    HairGeomG g = hair_geom(m, wo, hh);
    float w[4];
    hair_ap_pdf(m, g, hh, w);
    float u = pcg_f32(rng);
    int pp = HAIR_PMAX;
    for (int i = 0; i <= HAIR_PMAX; i++) {
        if (u < w[i]) { pp = i; break; }
        u -= w[i];
    }
    float v = m.hair_v[pp];
    float u1 = max(pcg_f32(rng), 1e-9f);
    float u2 = pcg_f32(rng);
    float cos_theta = 1.0f + v * log(u1 + (1.0f - u1) * exp(-2.0f / v));
    float sin_theta = sqrt(max(1.0f - cos_theta * cos_theta, 0.0f));
    float cpl = cos(2.0f * PT_PI * u2);
    float sti = -cos_theta * g.sin_to + sin_theta * cpl * g.cos_to;
    float cti = sqrt(max(1.0f - sti * sti, 0.0f));
    float u3 = pcg_f32(rng);
    float dphi = pp < HAIR_PMAX
        ? hair_phi_fn(pp, g.gamma_o, g.gamma_t) + hair_sample_trimmed_logistic(u3, m.hair_s)
        : 2.0f * PT_PI * u3;
    float phi_i = g.phi_o + dphi;
    float3 wi = float3(sti, cti * cos(phi_i), cti * sin(phi_i));
    float pv = hair_pdf(m, wo, wi, hh);
    if (pv <= 1e-12f) return false;
    wi_out = wi;
    f_out = hair_f(m, wo, wi, hh);
    pdf_out = pv;
    return true;
}

// Fiber frame from tangent/normal/viewer (mirror of hair::FiberFrame).
inline FrameL fiber_frame(float3 tangent, float3 n, float3 wo, thread float& h_out) {
    float3 x = normalize_cpu(tangent);
    float3 z = wo - x * dot(wo, x);
    if (dot(z, z) < 1e-12f) z = n - x * dot(n, x);
    z = normalize_cpu(z);
    float3 y = normalize_cpu(cross(z, x));
    float3 n_perp = normalize_cpu(n - x * dot(n, x));
    h_out = clamp(dot(n_perp, y), -1.0f, 1.0f);
    FrameL f;
    f.t = x;
    f.b = y;
    f.n = z;
    return f;
}

// ---- many-light sampling: stochastic light-BVH descent ------------------
// Mirrors scene::light_sampler (importance = power / max(d^2, r^2)).

inline float ls_importance(device const LightBvhNodeG& n, float3 p) {
    float3 c = float3(n.mn[0] + n.mx[0], n.mn[1] + n.mx[1], n.mn[2] + n.mx[2]) * 0.5f;
    float3 d = p - c;
    float d2 = dot(d, d);
    float3 r = float3(n.mx[0] - n.mn[0], n.mx[1] - n.mn[1], n.mx[2] - n.mn[2]) * 0.5f;
    float r2 = dot(r, r);
    return n.power / max(max(d2, r2), 1e-6f);
}

// Pick a light for NEE; returns light index, writes the selection pmf.
inline uint ls_sample(thread const PtScene& s, float3 p, float u,
                      thread float& pmf_out) {
    constant PtUniforms& un = *s.u;
    bool has_finite = un.light_bvh_count > 0u;
    bool has_infinite = un.infinite_total > 0.0f;
    float pmf = 1.0f;
    bool pick_infinite = has_infinite;
    if (has_finite && has_infinite) {
        if (u < un.p_infinite) {
            u /= un.p_infinite;
            pmf *= un.p_infinite;
        } else {
            u = (u - un.p_infinite) / (1.0f - un.p_infinite);
            pmf *= 1.0f - un.p_infinite;
            pick_infinite = false;
        }
    }
    if (pick_infinite) {
        float target = u * un.infinite_total;
        float acc = 0.0f;
        uint last = 0u;
        for (uint i = 0u; i < un.light_count; i++) {
            float w = s.light_aux[i].inf_weight;
            if (w <= 0.0f) continue;
            acc += w;
            last = i;
            if (target <= acc) {
                pmf_out = pmf * w / max(un.infinite_total, 1e-12f);
                return i;
            }
        }
        pmf_out = pmf * s.light_aux[last].inf_weight / max(un.infinite_total, 1e-12f);
        return last;
    }
    uint node = 0u;
    for (int guard = 0; guard < 64; guard++) {
        device const LightBvhNodeG& nd = s.light_bvh[node];
        if (nd.b == 0xFFFFFFFFu) {
            pmf_out = pmf;
            return nd.a;
        }
        float ia = ls_importance(s.light_bvh[nd.a], p);
        float ib = ls_importance(s.light_bvh[nd.b], p);
        float total = ia + ib;
        float pa = total > 1e-30f ? ia / total : 0.5f;
        if (u < pa) {
            u = min(u / pa, 1.0f - 1e-7f);
            pmf *= pa;
            node = nd.a;
        } else {
            u = min((u - pa) / (1.0f - pa), 1.0f - 1e-7f);
            pmf *= 1.0f - pa;
            node = nd.b;
        }
    }
    pmf_out = pmf;
    return s.light_bvh[node].a;
}

// Selection probability of `light` from p (MIS on emitter hits).
inline float ls_pmf(thread const PtScene& s, float3 p, uint light) {
    constant PtUniforms& un = *s.u;
    bool has_finite = un.light_bvh_count > 0u;
    float infw = s.light_aux[light].inf_weight;
    if (infw > 0.0f) {
        float group = has_finite ? un.p_infinite : 1.0f;
        return group * infw / max(un.infinite_total, 1e-12f);
    }
    uint leaf = s.light_aux[light].leaf;
    if (leaf == 0xFFFFFFFFu) return 0.0f;
    float pmf = un.infinite_total > 0.0f ? 1.0f - un.p_infinite : 1.0f;
    uint node = leaf;
    for (int guard = 0; guard < 64; guard++) {
        uint parent = s.light_bvh[node].parent;
        if (parent == 0xFFFFFFFFu) break;
        device const LightBvhNodeG& pn = s.light_bvh[parent];
        float ia = ls_importance(s.light_bvh[pn.a], p);
        float ib = ls_importance(s.light_bvh[pn.b], p);
        float total = ia + ib;
        float mine = (node == pn.a) ? ia : ib;
        pmf *= total > 1e-30f ? mine / total : 0.5f;
        node = parent;
    }
    return pmf;
}

// ---- unified BSDF evaluation for light sampling -------------------------
// Returns the BSDF value with any cosine already applied (surfaces:
// f * cos+; hair: f alone) and the solid-angle pdf for MIS.

struct EvalCtx {
    bool hair;
    bool phase;        // volume scatter point: HG phase in world space
    float phase_g;
    FrameL frame;      // surface frame (n = z) or fiber frame (t = x)
    float3 wo_l;       // local wo (surfaces/hair) or WORLD wo (phase)
    PtMaterial m;
    float eta;
    float h;
};

inline float3 eval_bsdf_weighted(thread const EvalCtx& c, float3 wi_world,
                                 thread float& pdf_out) {
    if (c.phase) {
        float ph = hg_phase_g(c.phase_g, -dot(c.wo_l, wi_world));
        pdf_out = ph;
        return float3(ph);
    }
    float3 wi_l = to_local(c.frame, wi_world);
    if (c.h > 1.5f && !c.hair) {
        // SSS exit vertex: white Lambert transmission (f*cos = cos/pi).
        if (wi_l.z <= 0.0f) { pdf_out = 0.0f; return float3(0.0f); }
        pdf_out = wi_l.z / PT_PI;
        return float3(wi_l.z / PT_PI);
    }
    if (c.hair) {
        pdf_out = hair_pdf(c.m, c.wo_l, wi_l, c.h);
        return hair_f(c.m, c.wo_l, wi_l, c.h);
    }
    if (wi_l.z <= 0.0f) {
        pdf_out = 0.0f;
        return float3(0.0f);
    }
    float3 f;
    bsdf_eval_pdf(c.m, c.wo_l, wi_l, c.eta, f, pdf_out);
    return f * max(wi_l.z, 0.0f);
}

inline float3 sample_one_light(thread const PtScene& s, PtLight l, float3 p,
                               float3 nbias, thread const EvalCtx& ec,
                               float pick_pmf, float time, uint medium,
                               thread Pcg32& rng) {
    float3 n = nbias;
    float3 rad = m3(l.radiance);

    if (l.kind == 0u) {                       // point
        float3 to_l = m3(l.a) - p;
        float dist2 = max(dot(to_l, to_l), 1e-12f);
        float dist = sqrt(dist2);
        float3 wi = to_l / dist;
        float pdf;
        float3 f = eval_bsdf_weighted(ec, wi, pdf);
        if (max(max(f.x, f.y), f.z) <= 0.0f) return float3(0.0f);
        float3 vis = pt_visibility(s, p, n, wi, dist, time, medium, rng);
        return f * rad * vis / (dist2 * pick_pmf);
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
        float pdf;
        float3 f = eval_bsdf_weighted(ec, wi, pdf);
        if (max(max(f.x, f.y), f.z) <= 0.0f) return float3(0.0f);
        float3 vis = pt_visibility(s, p, n, wi, PT_BIG, time, medium, rng);
        return f * rad * vis / pick_pmf;
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
        float cl = fabs(dot(wi, m3(l.normal)));
        if (cl < 1e-9f || l.area <= 0.0f) return float3(0.0f);
        float bp;
        float3 f = eval_bsdf_weighted(ec, wi, bp);
        if (max(max(f.x, f.y), f.z) <= 0.0f) return float3(0.0f);
        float3 vis = pt_visibility(s, p, n, wi, dist, time, medium, rng);
        if (max(max(vis.x, vis.y), vis.z) <= 0.0f) return float3(0.0f);
        float pdf_sa = pick_pmf * dist2 / (cl * l.area);
        float w = power_heuristic(pdf_sa, bp);
        return f * rad * vis * (w / pdf_sa);
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
        float bp;
        float3 f = eval_bsdf_weighted(ec, wi, bp);
        if (max(max(f.x, f.y), f.z) <= 0.0f) return float3(0.0f);
        float dist = sqrt(dist2) * ct
            - sqrt(max(radius * radius - dist2 * st * st, 0.0f));
        float3 vis = pt_visibility(s, p, n, wi, dist, time, medium, rng);
        if (max(max(vis.x, vis.y), vis.z) <= 0.0f) return float3(0.0f);
        float pdf_sa = pick_pmf / max(2.0f * PT_PI * (1.0f - cm), 1e-12f);
        float w = power_heuristic(pdf_sa, bp);
        return f * rad * vis * (w / pdf_sa);
    }
    // dome
    float3 wi, er;
    float pdf_sa;
    env_sample(s, rng, wi, er, pdf_sa);
    if (pdf_sa <= 0.0f) return float3(0.0f);
    pdf_sa *= pick_pmf;
    float bp;
    float3 f = eval_bsdf_weighted(ec, wi, bp);
    if (max(max(f.x, f.y), f.z) <= 0.0f) return float3(0.0f);
    float3 vis = pt_visibility(s, p, n, wi, PT_BIG, time, medium, rng);
    if (max(max(vis.x, vis.y), vis.z) <= 0.0f) return float3(0.0f);
    float w = power_heuristic(pdf_sa, bp);
    return f * (er * rad) * vis * (w / pdf_sa);
}

// ---- the path integrator (mirror of pt::trace) --------------------------

// aux layout: 0-2 albedo, 3-5 normal, 6 depth, 7 id, 8-10 diffuse share,
// 11 hit flag. Mirrors pt::trace_full's AuxSample.
inline float3 pt_li(thread const PtScene& s, float3 origin, float3 dir,
                    float time, thread Pcg32& rng, thread float* aux) {
    constant PtUniforms& u = *s.u;
    float3 l = float3(0.0f);
    float3 beta = float3(1.0f);
    float prev_pdf = 0.0f;
    float3 prev_origin = origin;
    bool from_camera = true;
    uint presence_skips = 0u;
    float num_lights = (float)u.light_count;
    // Ray cone for texture mip selection (mirror of pt::trace).
    float cone_width = 0.0f;
    float cone_spread = 2.0f * u.half_height / (float)u.height;
    // Participating medium the ray currently travels in.
    uint medium = u.atmosphere;
    int first_spec = -1;      // -1 unset, 0 diffuse-like, 1 specular-like
    bool aux_set = false;
    float travel = 0.0f;

    uint depth = 0u;
    while (depth < u.max_bounces) {
        PtHit h = pt_trace_scene(s, origin, dir, time);

        // Medium interaction along this segment (mirror of pt::trace).
        if (medium != 0xFFFFFFFFu) {
            device const MediumG& med = s.media[medium];
            float t_limit = h.hit ? h.t : PT_BIG;
            float mt;
            float3 mweight;
            if (med_sample_distance(med, origin, dir, t_limit, beta, rng, mt, mweight)) {
                beta *= mweight;
                float3 p = origin + dir * mt;
                if (!aux_set) {
                    aux_set = true;
                    float3 stv = float3(med.sigma_a[0] + med.sigma_s[0],
                                        med.sigma_a[1] + med.sigma_s[1],
                                        med.sigma_a[2] + med.sigma_s[2]);
                    aux[0] = med.sigma_s[0] / max(stv.x, 1e-9f);
                    aux[1] = med.sigma_s[1] / max(stv.y, 1e-9f);
                    aux[2] = med.sigma_s[2] / max(stv.z, 1e-9f);
                    float3 nv = -normalize_cpu(dir);
                    aux[3] = nv.x; aux[4] = nv.y; aux[5] = nv.z;
                    aux[6] = travel + mt;
                    aux[7] = 0.0f;
                    aux[11] = 1.0f;
                }
                if (first_spec < 0) first_spec = 0;
                float3 emv = float3(med.emission[0], med.emission[1], med.emission[2]);
                if (emv.x + emv.y + emv.z > 0.0f) l += beta * emv;
                float3 wo = -normalize_cpu(dir);
                if (u.light_count > 0u) {
                    EvalCtx ec;
                    ec.hair = false;
                    ec.phase = true;
                    ec.phase_g = med.g;
                    ec.frame = frame_of(float3(0.0f, 0.0f, 1.0f));
                    ec.wo_l = wo; // world-space wo for the phase path
                    ec.m = s.materials[0];
                    ec.eta = 1.0f;
                    ec.h = 0.0f;
                    float pick;
                    uint li = ls_sample(s, p, pcg_f32(rng), pick);
                    float3 c = sample_one_light(s, s.lights[li], p, float3(0.0f), ec,
                                                pick, time, medium, rng);
                    { float3 _c = beta * c; l += _c; if (first_spec != 1) { aux[8] += _c.x; aux[9] += _c.y; aux[10] += _c.z; } }
                }
                float ppdf;
                float3 wi = hg_sample_g(med.g, wo, pcg_f32(rng), pcg_f32(rng), ppdf);
                prev_pdf = ppdf;
                prev_origin = p;
                from_camera = false;
                cone_spread += 0.4f;
                origin = p;
                dir = wi;
                depth++;
                if (depth >= u.rr_start) {
                    float q = min(max(max(beta.x, beta.y), beta.z), 0.95f);
                    if (q <= 0.0f || pcg_f32(rng) > q) break;
                    beta /= q;
                }
                continue;
            }
            beta *= mweight;
            if (max(max(beta.x, beta.y), beta.z) <= 0.0f) break;
        }

        if (!h.hit) {
            if (u.dome_index != 0xFFFFFFFFu) {
                float3 d = normalize_cpu(dir);
                float w = 1.0f;
                if (!from_camera) {
                    float pl = env_pdf(s, d) * ls_pmf(s, prev_origin, u.dome_index);
                    w = power_heuristic(prev_pdf, pl);
                }
                float3 dome_rad = env_eval(s, d) * m3(s.lights[u.dome_index].radiance);
                { float3 _c = beta * dome_rad * w; l += _c; if (first_spec != 1) { aux[8] += _c.x; aux[9] += _c.y; aux[10] += _c.z; } }
            } else {
                { float3 _c = beta * c3(u.background); l += _c; if (first_spec != 1) { aux[8] += _c.x; aux[9] += _c.y; aux[10] += _c.z; } }
            }
            break;
        }
        PtMaterial m = s.materials[h.material];

        // Invisible volume hull: crossing toggles the medium.
        if (is_volume_hull(m) && presence_skips < 16u) {
            presence_skips++;
            travel += h.t;
            medium = h.front ? m.interior : u.atmosphere;
            origin = h.p + normalize_cpu(dir) * PT_RAY_OFFSET;
            continue;
        }

        float3 wo = -normalize_cpu(dir);
        cone_width += h.t * cone_spread;
        travel += h.t;
        apply_patterns(h.material, m, h.st, h.p, h.n,
                       cone_width * h.st_density, s.tex_data, s.tex_mips);
        if (!aux_set) {
            aux_set = true;
            float lum_f0 = 0.2126f * m.spec_f0[0] + 0.7152f * m.spec_f0[1]
                + 0.0722f * m.spec_f0[2];
            float has_spec = lum_f0 > 1e-6f ? 1.0f : 0.0f;
            for (int k = 0; k < 3; k++) {
                aux[k] = min(m.diffuse_color[k] * m.diffuse_gain
                    + m.spec_f0[k] * has_spec
                    + m.refr_color[k] * m.glass_gain, 1.0f);
            }
            float3 nv = dot(h.n, wo) >= 0.0f ? h.n : -h.n;
            aux[3] = nv.x; aux[4] = nv.y; aux[5] = nv.z;
            aux[6] = travel;
            aux[7] = (float)m.obj_id;
            aux[11] = 1.0f;
        }

        float presence = clamp(m.presence, 0.0f, 1.0f);
        if (presence < 1.0f && pcg_f32(rng) >= presence && presence_skips < 16u) {
            presence_skips++;
            travel += h.t;
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
                    * ls_pmf(s, prev_origin, m.area_light);
                w = power_heuristic(prev_pdf, pl);
            }
            { float3 _c = beta * em * w; l += _c; if (first_spec != 1) { aux[8] += _c.x; aux[9] += _c.y; aux[10] += _c.z; } }
        }

        // Hair fibers: full-sphere Marschner scattering (mirror of the
        // CPU hair branch in pt::trace).
        if (m.is_hair != 0u && dot(h.tangent, h.tangent) > 0.5f) {
            float hh;
            FrameL ff = fiber_frame(h.tangent, h.n, wo, hh);
            float3 wo_f = to_local(ff, wo);
            if (u.light_count > 0u) {
                EvalCtx ec;
                ec.hair = true;
                ec.phase = false;
                ec.phase_g = 0.0f;
                ec.frame = ff;
                ec.wo_l = wo_f;
                ec.m = m;
                ec.eta = m.hair_eta;
                ec.h = hh;
                float pick;
                uint li = ls_sample(s, h.p, pcg_f32(rng), pick);
                float3 c = sample_one_light(s, s.lights[li], h.p, n, ec, pick, time, medium, rng);
                { float3 _c = beta * c; l += _c; if (first_spec != 1) { aux[8] += _c.x; aux[9] += _c.y; aux[10] += _c.z; } }
            }
            float3 wi_f, fv;
            float pv;
            if (!hair_sample(m, wo_f, hh, rng, wi_f, fv, pv)) break;
            float3 wi = to_world(ff, wi_f);
            beta *= fv / pv;   // per-solid-angle f: no cosine
            if (first_spec < 0) first_spec = 0;
            prev_pdf = pv;
            prev_origin = h.p;
            from_camera = false;
            cone_spread += 0.4f;
            float3 side = dot(wi, h.n) >= 0.0f ? h.n : -h.n;
            origin = h.p + side * PT_RAY_OFFSET;
            dir = wi;
            depth++;
            if (depth >= u.rr_start) {
                float q = min(max(max(beta.x, beta.y), beta.z), 0.95f);
                if (q <= 0.0f || pcg_f32(rng) > q) break;
                beta /= q;
            }
            continue;
        }

        // Subsurface random walk (mirror of pt::trace's SSS branch).
        if (m.sss_gain > 0.0f && m.is_hair == 0u && pcg_f32(rng) < m.sss_gain) {
            if (first_spec < 0) first_spec = 0;
            MediumG sm;
            for (int k = 0; k < 3; k++) {
                sm.sigma_s[k] = m.sss_sigma_s[k];
                sm.sigma_a[k] = m.sss_sigma_t[k] - m.sss_sigma_s[k];
            }
            sm.g = 0.0f;
            sm.has_density = 0u;
            sm.majorant = 0.0f;
            // Cosine entry around -n.
            FrameL ef = frame_of(-n);
            float eu1 = pcg_f32(rng);
            float eu2 = pcg_f32(rng);
            float er = sqrt(eu1);
            float eph = 2.0f * PT_PI * eu2;
            float3 wdir = to_world(ef, float3(er * cos(eph), er * sin(eph),
                                              sqrt(max(1.0f - eu1, 0.0f))));
            float3 wpos = h.p - n * PT_RAY_OFFSET;
            bool exited = false;
            for (int step = 0; step < 256; step++) {
                PtHit wh = pt_trace_scene(s, wpos, wdir, time);
                if (!wh.hit) break;
                float mt;
                float3 mweight;
                // Local copy: device-const media functions need a device ref;
                // build inline homogeneous sampling instead.
                float3 st3 = float3(m.sss_sigma_t[0], m.sss_sigma_t[1], m.sss_sigma_t[2]);
                float3 ss3 = float3(m.sss_sigma_s[0], m.sss_sigma_s[1], m.sss_sigma_s[2]);
                {
                    float3 b = max(beta, float3(0.0f));
                    float total = b.x + b.y + b.z;
                    float3 w = total > 1e-12f ? b / total : float3(1.0f / 3.0f);
                    float u1 = pcg_f32(rng);
                    float sigma_c = u1 < w.x ? st3.x : (u1 < w.x + w.y ? st3.y : st3.z);
                    sigma_c = max(sigma_c, 1e-9f);
                    float t = -log(max(1.0f - pcg_f32(rng), 1e-12f)) / sigma_c;
                    if (t < wh.t) {
                        float3 tr = exp(-st3 * t);
                        float pdf = w.x * st3.x * tr.x + w.y * st3.y * tr.y
                            + w.z * st3.z * tr.z;
                        if (pdf <= 1e-30f) break;
                        mweight = ss3 * tr / pdf;
                        mt = t;
                        // Scatter: isotropic direction.
                        beta *= mweight;
                        float3 sp = wpos + wdir * mt;
                        float su = pcg_f32(rng);
                        float sv = pcg_f32(rng);
                        float z = 1.0f - 2.0f * su;
                        float rr2 = sqrt(max(1.0f - z * z, 0.0f));
                        float sph = 2.0f * PT_PI * sv;
                        wpos = sp;
                        wdir = float3(rr2 * cos(sph), rr2 * sin(sph), z);
                        float q = min(max(max(beta.x, beta.y), beta.z), 0.95f);
                        if (q <= 0.0f || (step > 8 && pcg_f32(rng) > q)) break;
                        if (step > 8) beta /= q;
                        continue;
                    }
                    // Pass to the boundary: exit diffusely.
                    float3 tr = exp(-st3 * wh.t);
                    float pdf = w.x * tr.x + w.y * tr.y + w.z * tr.z;
                    if (pdf <= 1e-30f) break;
                    beta *= tr / pdf;
                }
                float3 out_n = dot(wh.n, wdir) > 0.0f ? wh.n : -wh.n;
                FrameL xf = frame_of(out_n);
                float3 exit_p = wpos + wdir * wh.t;
                if (u.light_count > 0u) {
                    EvalCtx ec;
                    ec.hair = false;
                    ec.phase = false;
                    ec.phase_g = 0.0f;
                    ec.frame = xf;
                    ec.wo_l = float3(0.0f, 0.0f, 1.0f);
                    ec.m = m;
                    ec.eta = 1.0f;
                    ec.h = 2.0f; // sentinel: lambert-exit eval
                    float pick;
                    uint li = ls_sample(s, exit_p, pcg_f32(rng), pick);
                    float3 c = sample_one_light(s, s.lights[li], exit_p, out_n, ec,
                                                pick, time, medium, rng);
                    { float3 _c = beta * c; l += _c; if (first_spec != 1) { aux[8] += _c.x; aux[9] += _c.y; aux[10] += _c.z; } }
                }
                float xu1 = pcg_f32(rng);
                float xu2 = pcg_f32(rng);
                float xr = sqrt(xu1);
                float xph = 2.0f * PT_PI * xu2;
                float3 lw = float3(xr * cos(xph), xr * sin(xph),
                                   sqrt(max(1.0f - xu1, 0.0f)));
                float3 wi = to_world(xf, lw);
                prev_pdf = max(lw.z, 1e-9f) / PT_PI;
                prev_origin = exit_p;
                from_camera = false;
                cone_spread += 0.4f;
                origin = exit_p + out_n * PT_RAY_OFFSET;
                dir = wi;
                exited = true;
                break;
            }
            if (!exited) break;
            depth++;
            if (depth >= u.rr_start) {
                float q = min(max(max(beta.x, beta.y), beta.z), 0.95f);
                if (q <= 0.0f || pcg_f32(rng) > q) break;
                beta /= q;
            }
            continue;
        }

        FrameL frame = frame_of(n);
        float3 wo_l = to_local(frame, wo);
        float wsum = m.weights[0] + m.weights[1] + m.weights[2] + m.weights[3]
            + m.weights[4];

        if (u.light_count > 0u && wsum > 0.0f) {
            EvalCtx ec;
            ec.hair = false;
            ec.phase = false;
            ec.phase_g = 0.0f;
            ec.frame = frame;
            ec.wo_l = wo_l;
            ec.m = m;
            ec.eta = eta;
            ec.h = 0.0f;
            float pick;
            uint li = ls_sample(s, h.p, pcg_f32(rng), pick);
            float3 c = sample_one_light(s, s.lights[li], h.p, n, ec, pick, time, medium, rng);
            { float3 _c = beta * c; l += _c; if (first_spec != 1) { aux[8] += _c.x; aux[9] += _c.y; aux[10] += _c.z; } }
        }

        float3 wi_l, f;
        float pdf;
        bool transmitted;
        bool spec_lobe;
        if (!bsdf_sample(m, wo_l, eta, rng, wi_l, f, pdf, transmitted, spec_lobe)) break;
        float3 wi = to_world(frame, wi_l);
        beta *= f * (fabs(wi_l.z) / pdf);
        if (first_spec < 0) first_spec = spec_lobe ? 1 : 0;
        prev_pdf = pdf;
        prev_origin = h.p;
        from_camera = false;
        cone_spread += min(1.0f / (1.0f + pdf), 0.4f);

        if (transmitted && m.interior != 0xFFFFFFFFu) {
            medium = entering ? m.interior : u.atmosphere;
        }
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

// Filter importance sampling: subpixel offset from one uniform,
// mirroring PixelFilter::sample_1d on the CPU.
inline float filter_offset_1d(uint kind, float width, float u) {
    if (kind == 1u) {                      // triangle
        float half_w = width * 0.5f;
        return u < 0.5f ? (sqrt(2.0f * u) - 1.0f) * half_w
                        : (1.0f - sqrt(2.0f * (1.0f - u))) * half_w;
    }
    if (kind == 2u) {                      // truncated gaussian
        float u1 = max(u, 1e-12f);
        float u2 = fract(u * 2654435761.0f);
        float sigma = width / 4.0f;
        float g = sqrt(-2.0f * log(u1)) * cos(2.0f * PT_PI * u2);
        return clamp(g * sigma, -width * 0.5f, width * 0.5f);
    }
    return (u - 0.5f) * width;             // box
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
                      device const float*      st               [[buffer(14)]],
                      device const float*      tex_data         [[buffer(15)]],
                      device const TexMipG*    tex_mips         [[buffer(16)]],
                      device const float*      vertices1        [[buffer(17)]],
                      device const CurveSegG*  curve_segs       [[buffer(18)]],
                      device const CurveInfoG* curve_infos      [[buffer(19)]],
                      device const LightBvhNodeG* light_bvh     [[buffer(20)]],
                      device const LightAuxG*  light_aux        [[buffer(21)]],
                      device const MediumG*    media            [[buffer(22)]],
                      constant PtUniforms&     u                [[buffer(23)]],
                      device float*            accum            [[buffer(24)]],
                      device float*            aux_accum        [[buffer(25)]],
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
    s.st = st;
    s.vertices1 = vertices1;
    s.curve_segs = curve_segs;
    s.curve_infos = curve_infos;
    s.light_bvh = light_bvh;
    s.light_aux = light_aux;
    s.media = media;
    s.tex_data = tex_data;
    s.tex_mips = tex_mips;
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
        // Camera sample: filter-importance-sampled subpixel offset, then
        // (optionally) lens position and shutter time — the same draw
        // order as pt::camera_ray on the CPU.
        float fu = pcg_f32(rng);
        float fv = pcg_f32(rng);
        float px = (float)gid.x + 0.5f + filter_offset_1d(u.filter_kind, u.filter_width, fu);
        float py = (float)py_row + 0.5f + filter_offset_1d(u.filter_kind, u.filter_width, fv);
        float uu = (px / (float)u.width) * 2.0f - 1.0f;
        float vv = 1.0f - (py / (float)u.height) * 2.0f;

        float3 ray_o;
        float3 d;
        if (u.projection == 1u) {          // orthographic
            ray_o = eye + rgt * (uu * u.ortho_half_w) + upv * (vv * u.ortho_half_h);
            d = fwd;
        } else {
            float3 dir = fwd + rgt * (uu * u.half_width) + upv * (vv * u.half_height);
            if (u.lens_radius > 0.0f) {
                float lu = pcg_f32(rng);
                float lv = pcg_f32(rng);
                float r = u.lens_radius * sqrt(lu);
                float phi = 2.0f * PT_PI * lv;
                float3 offset = rgt * (r * cos(phi)) + upv * (r * sin(phi));
                float3 focus = eye + dir * u.focal_distance;
                ray_o = eye + offset;
                d = normalize_cpu(focus - ray_o);
            } else {
                ray_o = eye;
                d = normalize_cpu(dir);
            }
        }
        float rtime = (u.has_motion != 0u) ? pcg_f32(rng) : 0.0f;
        float aux[12];
        for (int k = 0; k < 12; k++) aux[k] = 0.0f;
        float3 li = pt_li(s, ray_o, d, rtime, rng, aux);
        float lum = 0.2126f * li.x + 0.7152f * li.y + 0.0722f * li.z;
        if (lum > u.firefly_clamp) {
            float kk = u.firefly_clamp / lum;
            li *= kk;
            aux[8] *= kk; aux[9] *= kk; aux[10] *= kk;
        }
        sum += li;
        uint aidx = (py_row * u.width + gid.x) * 12u;
        for (int k = 0; k < 7; k++) aux_accum[aidx + k] += aux[k];
        // id: first writer wins (deterministic: sample 0 lands first).
        if (aux_accum[aidx + 11u] == 0.0f && aux[11] > 0.0f) {
            aux_accum[aidx + 7u] = aux[7];
        }
        for (int k = 8; k < 11; k++) aux_accum[aidx + k] += aux[k];
        aux_accum[aidx + 11u] += aux[11];
    }

    uint idx = (py_row * u.width + gid.x) * 4u;
    accum[idx + 0u] += sum.x;
    accum[idx + 1u] += sum.y;
    accum[idx + 2u] += sum.z;
    accum[idx + 3u] += (float)u.sample_count;
}
