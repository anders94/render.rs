// Wavefront scheduler (the P9-measured deferral): instead of one
// megakernel thread owning a whole path, the path state lives in device
// buffers and three small kernels advance ALL live paths in lockstep:
//
//   wf_raygen  — camera samples -> initial path states (identity queue)
//   wf_extend  — closest-hit for every queued path (coherent traversal,
//                small register footprint -> high occupancy)
//   wf_shade   — pt_shade_step (the exact megakernel step function) on
//                the stored hit; survivors are compacted into the next
//                queue via an atomic counter, dead paths splat into the
//                accumulation buffer
//
// The win is structural: after Russian roulette starts killing paths,
// the megakernel's warps carry dead lanes to the bitter end — the
// wavefront queues shrink instead. Sampling streams are identical to the
// megakernel (same per-(pixel,sample) PCG32, same step function), so the
// two schedulers converge to the same image.
//
// Compiled after kernel_pt.metal; reuses PtScene, WfState, pt_shade_step,
// pt_trace_scene, filter_offset_1d and friends.

struct WfPath {
    float4 o;        // origin.xyz, time
    float4 d;        // dir.xyz, prev_pdf
    float4 beta;     // beta.xyz, cone_width
    float4 l;        // l.xyz, cone_spread
    float4 prev;     // prev_origin.xyz, travel
    uint4 rng;       // pcg state (lo, hi), inc (lo, hi)
    // med stack packed 8 x u8 (0xFF empty), flags, pixel index, spare
    uint4 misc;      // [0]=med lo, [1]=med hi | flags<<?, see pack/unpack
    uint4 misc2;     // [0]=flags, [1]=pixel, [2]=med_depth, [3]=pad
};

struct WfHitRec {
    float4 p_t;      // p.xyz, t
    float4 n_front;  // n.xyz, front (1/0)
    float4 st_den;   // st.xy, st_density, hit (1/0)
    float4 tangent;  // tangent.xyz, material-as-float-bits
};

inline void wf_pack(thread const WfState& st, uint pixel, thread const Pcg32& rng,
                    device WfPath* paths, uint i) {
    WfPath p;
    p.o = float4(st.origin, st.time);
    p.d = float4(st.dir, st.prev_pdf);
    p.beta = float4(st.beta, st.cone_width);
    p.l = float4(st.l, st.cone_spread);
    p.prev = float4(st.prev_origin, st.travel);
    p.rng = uint4((uint)(rng.state & 0xFFFFFFFFUL), (uint)(rng.state >> 32),
                  (uint)(rng.inc & 0xFFFFFFFFUL), (uint)(rng.inc >> 32));
    uint med_lo = 0u;
    uint med_hi = 0u;
    for (uint k = 0u; k < 4u; k++) {
        med_lo |= ((k < st.med_depth ? st.med_stack[k] : 0xFFu) & 0xFFu) << (8u * k);
    }
    for (uint k = 0u; k < 4u; k++) {
        uint idx = k + 4u;
        med_hi |= ((idx < st.med_depth ? st.med_stack[idx] : 0xFFu) & 0xFFu) << (8u * k);
    }
    p.misc = uint4(med_lo, med_hi, 0u, 0u);
    uint flags = (st.from_camera ? 1u : 0u)
        | (st.aux_set ? 2u : 0u)
        | ((uint)(st.first_spec + 1) << 2)       // 2 bits
        | ((st.presence_skips & 0x1Fu) << 4)
        | ((st.depth & 0xFFu) << 9);
    p.misc2 = uint4(flags, pixel, st.med_depth, 0u);
    paths[i] = p;
}

inline void wf_unpack(device const WfPath* paths, uint i, thread WfState& st,
                      thread uint& pixel, thread Pcg32& rng) {
    WfPath p = paths[i];
    st.origin = p.o.xyz;
    st.time = p.o.w;
    st.dir = p.d.xyz;
    st.prev_pdf = p.d.w;
    st.beta = p.beta.xyz;
    st.cone_width = p.beta.w;
    st.l = p.l.xyz;
    st.cone_spread = p.l.w;
    st.prev_origin = p.prev.xyz;
    st.travel = p.prev.w;
    rng.state = (ulong)p.rng.x | ((ulong)p.rng.y << 32);
    rng.inc = (ulong)p.rng.z | ((ulong)p.rng.w << 32);
    st.med_depth = min(p.misc2.z, 8u);
    for (uint k = 0u; k < 8u; k++) {
        uint byte = k < 4u ? (p.misc.x >> (8u * k)) & 0xFFu
                           : (p.misc.y >> (8u * (k - 4u))) & 0xFFu;
        st.med_stack[k] = byte;
    }
    uint flags = p.misc2.x;
    st.from_camera = (flags & 1u) != 0u;
    st.aux_set = (flags & 2u) != 0u;
    st.first_spec = (int)((flags >> 2) & 3u) - 1;
    st.presence_skips = (flags >> 4) & 0x1Fu;
    st.depth = (flags >> 9) & 0xFFu;
    pixel = p.misc2.y;
}

#define WF_SCENE_ARGS \
    device const Object*     objects          [[buffer(0)]], \
    device const uint*       object_materials [[buffer(1)]], \
    device const PtMaterial* materials        [[buffer(2)]], \
    device const PtLight*    lights           [[buffer(3)]], \
    device const BvhNodeG*   tlas             [[buffer(4)]], \
    device const InstanceG*  instances        [[buffer(5)]], \
    device const BvhNodeG*   blas             [[buffer(6)]], \
    device const uint*       tri_indices      [[buffer(7)]], \
    device const float*      vertices         [[buffer(8)]], \
    device const float*      normals          [[buffer(9)]], \
    device const MeshInfoG*  mesh_infos       [[buffer(10)]], \
    device const float*      env_pixels       [[buffer(11)]], \
    device const float*      env_marginal     [[buffer(12)]], \
    device const float*      env_conditional  [[buffer(13)]], \
    device const float*      st_buf           [[buffer(14)]], \
    device const float*      tex_data         [[buffer(15)]], \
    device const TexMipG*    tex_mips         [[buffer(16)]], \
    device const float*      vertices1        [[buffer(17)]], \
    device const CurveSegG*  curve_segs       [[buffer(18)]], \
    device const CurveInfoG* curve_infos      [[buffer(19)]], \
    device const LightBvhNodeG* light_bvh     [[buffer(20)]], \
    device const LightAuxG*  light_aux        [[buffer(21)]], \
    device const MediumG*    media            [[buffer(22)]], \
    constant PtUniforms&     u                [[buffer(23)]]

#define WF_FILL_SCENE(s) \
    s.objects = objects; \
    s.object_materials = object_materials; \
    s.materials = materials; \
    s.lights = lights; \
    s.tlas = tlas; \
    s.instances = instances; \
    s.blas = blas; \
    s.tri_indices = tri_indices; \
    s.vertices = vertices; \
    s.normals = normals; \
    s.mesh_infos = mesh_infos; \
    s.st = st_buf; \
    s.vertices1 = vertices1; \
    s.curve_segs = curve_segs; \
    s.curve_infos = curve_infos; \
    s.light_bvh = light_bvh; \
    s.light_aux = light_aux; \
    s.media = media; \
    s.tex_data = tex_data; \
    s.tex_mips = tex_mips; \
    s.env_pixels = env_pixels; \
    s.env_marginal = env_marginal; \
    s.env_conditional = env_conditional; \
    s.u = &u;

// Camera generation for path `gid + y_offset*width`-style chunking: the
// host passes chunk_start in u.y_offset and the sample index in
// u.sample_start. One path per pixel per wave.
kernel void wf_raygen(WF_SCENE_ARGS,
                      device WfPath*  paths  [[buffer(24)]],
                      device WfHitRec* hits  [[buffer(25)]],
                      device uint* q_in      [[buffer(26)]],
                      device atomic_uint* q_out [[buffer(27)]],
                      uint tid [[thread_position_in_grid]]) {
    // tid is slab-local; u.y_offset carries the slab's first pixel id.
    uint path_id = tid + u.y_offset;
    uint total = u.width * u.height;
    if (path_id >= total) return;
    PtScene s;
    WF_FILL_SCENE(s)

    uint px = path_id % u.width;
    uint py = path_id / u.width;
    Pcg32 rng = pcg_for_pixel_sample((ulong)path_id, (ulong)u.sample_start);

    float3 eye = c3(u.eye);
    float3 fwd = c3(u.forward);
    float3 rgt = c3(u.right);
    float3 upv = c3(u.up);
    float fu = pcg_f32(rng);
    float fv = pcg_f32(rng);
    float pxf = (float)px + 0.5f + filter_offset_1d(u.filter_kind, u.filter_width, fu);
    float pyf = (float)py + 0.5f + filter_offset_1d(u.filter_kind, u.filter_width, fv);
    float uu = (pxf / (float)u.width) * 2.0f - 1.0f;
    float vv = 1.0f - (pyf / (float)u.height) * 2.0f;
    float3 ray_o;
    float3 d;
    if (u.projection == 1u) {
        ray_o = eye + rgt * (uu * u.ortho_half_w) + upv * (vv * u.ortho_half_h);
        d = fwd;
    } else {
        float3 dir0 = fwd + rgt * (uu * u.half_width) + upv * (vv * u.half_height);
        if (u.lens_radius > 0.0f) {
            float lu = pcg_f32(rng);
            float lv = pcg_f32(rng);
            float r = u.lens_radius * sqrt(lu);
            float phi = 2.0f * PT_PI * lv;
            float3 offset = rgt * (r * cos(phi)) + upv * (r * sin(phi));
            float3 focus = eye + dir0 * u.focal_distance;
            ray_o = eye + offset;
            d = normalize_cpu(focus - ray_o);
        } else {
            ray_o = eye;
            d = normalize_cpu(dir0);
        }
    }
    float rtime = (u.has_motion != 0u) ? pcg_f32(rng) : 0.0f;
    if (u.has_cam_motion != 0u && rtime > 0.0f) {
        Affine mi = load_affine_c(u.cam_motion_inv);
        Affine id;
        id.r0 = float4(1.0f, 0.0f, 0.0f, 0.0f);
        id.r1 = float4(0.0f, 1.0f, 0.0f, 0.0f);
        id.r2 = float4(0.0f, 0.0f, 1.0f, 0.0f);
        Affine mt = affine_lerp(id, mi, rtime);
        ray_o = xf_point(mt, ray_o);
        d = normalize_cpu(xf_vec(mt, d));
    }

    WfState st;
    wf_state_init(st, s, ray_o, d, rtime);
    uint local = path_id - u.wf_slab_base;
    wf_pack(st, path_id, rng, paths, local);
    q_in[local] = local;
    (void)hits;
    (void)q_out;
}

// Closest hit for queued paths. u.y_offset = chunk start into the queue,
// u.sample_count = live count.
kernel void wf_extend(WF_SCENE_ARGS,
                      device WfPath*  paths  [[buffer(24)]],
                      device WfHitRec* hits  [[buffer(25)]],
                      device const uint* q_in [[buffer(26)]],
                      device atomic_uint* q_out [[buffer(27)]],
                      uint tid [[thread_position_in_grid]]) {
    uint qi = tid + u.y_offset;
    if (qi >= u.sample_count) return;
    PtScene s;
    WF_FILL_SCENE(s)
    uint path_id = q_in[qi];
    WfPath p = paths[path_id];
    PtHit h = pt_trace_scene(s, p.o.xyz, p.d.xyz, p.o.w);
    WfHitRec r;
    r.p_t = float4(h.p, h.t);
    r.n_front = float4(h.n, h.front ? 1.0f : 0.0f);
    r.st_den = float4(h.st.x, h.st.y, h.st_density, h.hit ? 1.0f : 0.0f);
    r.tangent = float4(h.tangent, as_type<float>(h.material));
    hits[path_id] = r;
    (void)q_out;
}

// Shade queued paths from their stored hits; survivors compact into
// q_out (q_out[0] is the atomic count, entries start at 1).
kernel void wf_shade(WF_SCENE_ARGS,
                     device WfPath*  paths  [[buffer(24)]],
                     device WfHitRec* hits  [[buffer(25)]],
                     device const uint* q_in [[buffer(26)]],
                     device atomic_uint* q_out [[buffer(27)]],
                     device float* accum     [[buffer(28)]],
                     uint tid [[thread_position_in_grid]]) {
    uint qi = tid + u.y_offset;
    if (qi >= u.sample_count) return;
    PtScene s;
    WF_FILL_SCENE(s)
    uint path_id = q_in[qi];

    WfState st;
    uint pixel;
    Pcg32 rng;
    wf_unpack(paths, path_id, st, pixel, rng);

    WfHitRec r = hits[path_id];
    PtHit h;
    h.hit = r.st_den.w > 0.5f;
    h.t = r.p_t.w;
    h.p = r.p_t.xyz;
    h.n = r.n_front.xyz;
    h.front = r.n_front.w > 0.5f;
    h.st = float2(r.st_den.x, r.st_den.y);
    h.st_density = r.st_den.z;
    h.tangent = r.tangent.xyz;
    h.material = as_type<uint>(r.tangent.w);

    float aux[12];
    for (int k = 0; k < 12; k++) aux[k] = 0.0f;
    int alive = pt_shade_step(s, st, h, rng, aux);
    if (alive == 1 && st.depth < u.max_bounces) {
        wf_pack(st, pixel, rng, paths, path_id);
        uint slot = atomic_fetch_add_explicit(q_out, 1u, memory_order_relaxed);
        // q_out entries live after the counter word.
        device uint* entries = (device uint*)q_out + 4;
        entries[slot] = path_id;
        return;
    }
    // Path finished: firefly clamp + splat (one path per pixel per wave —
    // plain adds are race-free).
    float3 li = st.l;
    float lum = 0.2126f * li.x + 0.7152f * li.y + 0.0722f * li.z;
    if (lum > u.firefly_clamp) li *= u.firefly_clamp / lum;
    uint idx = pixel * 4u;
    accum[idx + 0u] += li.x;
    accum[idx + 1u] += li.y;
    accum[idx + 2u] += li.z;
    accum[idx + 3u] += 1.0f;
}
