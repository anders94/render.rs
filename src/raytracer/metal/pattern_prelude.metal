// Pattern-graph runtime support (roadmap Phase 6). Compiled between
// isect_common and the generated pat_node_* functions; kernel_pt.metal
// calls apply_patterns (always emitted by the codegen, possibly a no-op).
//
// Texture storage: one flat float buffer holding every referenced
// texture's full mip chain, row-major RGB, with a TexMipG entry per
// (texture, level). Sampling mirrors src/texture/cache.rs bilinear /
// trilinear exactly.

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
    uint  is_hair;          // Marschner hair fields below
    float hair_sigma_a[3];
    float hair_v[4];        // longitudinal variances (R, TT, TRT, residual)
    float hair_s;           // azimuthal logistic scale
    float hair_eta;
    uint  hair_pad;
};

struct TexMipG {
    uint offset;   // float index of first channel
    uint width;
    uint height;
    uint pad;
};

// Wrap codes: 0 periodic, 1 clamp, 2 black.
inline float3 tex_texel(device const float* td, device const TexMipG* tm,
                        uint mip_index, int x, int y, uint wrap) {
    TexMipG info = tm[mip_index];
    int w = (int)info.width;
    int h = (int)info.height;
    if (wrap == 0u) {
        x = ((x % w) + w) % w;
        y = ((y % h) + h) % h;
    } else if (wrap == 1u) {
        x = clamp(x, 0, w - 1);
        y = clamp(y, 0, h - 1);
    } else {
        if (x < 0 || x >= w || y < 0 || y >= h) return float3(0.0f);
    }
    uint i = info.offset + ((uint)y * info.width + (uint)x) * 3u;
    return float3(td[i], td[i + 1u], td[i + 2u]);
}

inline float3 tex_bilinear(device const float* td, device const TexMipG* tm,
                           uint mip_index, float s, float t, uint wrap) {
    TexMipG info = tm[mip_index];
    float fx = s * (float)info.width - 0.5f;
    float fy = t * (float)info.height - 0.5f;
    float x0f = floor(fx);
    float y0f = floor(fy);
    float ax = fx - x0f;
    float ay = fy - y0f;
    int x0 = (int)x0f;
    int y0 = (int)y0f;
    float3 c00 = tex_texel(td, tm, mip_index, x0, y0, wrap);
    float3 c10 = tex_texel(td, tm, mip_index, x0 + 1, y0, wrap);
    float3 c01 = tex_texel(td, tm, mip_index, x0, y0 + 1, wrap);
    float3 c11 = tex_texel(td, tm, mip_index, x0 + 1, y0 + 1, wrap);
    return c00 * (1.0f - ax) * (1.0f - ay) + c10 * ax * (1.0f - ay)
         + c01 * (1.0f - ax) * ay + c11 * ax * ay;
}

inline float3 tex_sample(device const float* td, device const TexMipG* tm,
                         uint mip_start, uint mip_count, uint base_w, uint base_h,
                         float s, float t, float footprint, uint wrap) {
    float base = (float)max(base_w, base_h);
    float texels = max(footprint * base, 1e-9f);
    float mip_f = clamp(log2(texels), 0.0f, (float)(mip_count - 1u));
    uint m0 = (uint)floor(mip_f);
    uint m1 = min(m0 + 1u, mip_count - 1u);
    float a = mip_f - (float)m0;
    float3 lo = tex_bilinear(td, tm, mip_start + m0, s, t, wrap);
    if (a < 1e-6f || m1 == m0) return lo;
    float3 hi = tex_bilinear(td, tm, mip_start + m1, s, t, wrap);
    return lo * (1.0f - a) + hi * a;
}

// ---- Perlin fBm over P, mirroring src/geometry/displace.rs (f32) --------

inline uint pat_hash64(long x) {
    x = x * (long)0x9e3779b97f4a7c15UL;
    x ^= (x >> 29);
    x = x * (long)0xbf58476d1ce4e5b9UL;
    x ^= (x >> 32);
    return (uint)x;
}

inline float3 pat_gradient(long ix, long iy, long iz) {
    uint h = pat_hash64(ix * 73856093L ^ iy * 19349663L ^ iz * 83492791L);
    switch (h % 12u) {
        case 0u:  return float3( 1.0f,  1.0f,  0.0f);
        case 1u:  return float3(-1.0f,  1.0f,  0.0f);
        case 2u:  return float3( 1.0f, -1.0f,  0.0f);
        case 3u:  return float3(-1.0f, -1.0f,  0.0f);
        case 4u:  return float3( 1.0f,  0.0f,  1.0f);
        case 5u:  return float3(-1.0f,  0.0f,  1.0f);
        case 6u:  return float3( 1.0f,  0.0f, -1.0f);
        case 7u:  return float3(-1.0f,  0.0f, -1.0f);
        case 8u:  return float3( 0.0f,  1.0f,  1.0f);
        case 9u:  return float3( 0.0f, -1.0f,  1.0f);
        case 10u: return float3( 0.0f,  1.0f, -1.0f);
        default:  return float3( 0.0f, -1.0f, -1.0f);
    }
}

inline float pat_fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

inline float pat_perlin(float3 p) {
    float3 cell = floor(p);
    float3 fr = p - cell;
    long ix = (long)cell.x;
    long iy = (long)cell.y;
    long iz = (long)cell.z;
    float c[8];
    for (int k = 0; k < 8; k++) {
        long dx = (long)(k & 1);
        long dy = (long)((k >> 1) & 1);
        long dz = (long)((k >> 2) & 1);
        float3 g = pat_gradient(ix + dx, iy + dy, iz + dz);
        float3 d = fr - float3((float)dx, (float)dy, (float)dz);
        c[k] = dot(g, d);
    }
    float u = pat_fade(fr.x);
    float v = pat_fade(fr.y);
    float w = pat_fade(fr.z);
    float x00 = mix(c[0], c[1], u);
    float x10 = mix(c[2], c[3], u);
    float x01 = mix(c[4], c[5], u);
    float x11 = mix(c[6], c[7], u);
    float y0 = mix(x00, x10, v);
    float y1 = mix(x01, x11, v);
    return mix(y0, y1, w);
}

inline float pat_fbm(float3 p, float frequency, uint octaves, float gain,
                     float lacunarity) {
    float sum = 0.0f;
    float amp = 1.0f;
    float freq = frequency;
    for (uint i = 0u; i < octaves; i++) {
        sum += amp * pat_perlin(p * freq);
        amp *= gain;
        freq *= lacunarity;
    }
    return sum;
}

// ---- derived-quantity recompute after pattern overrides -----------------
// Mirrors GpuPtScene::build's material export (F0 layering + lobe weights,
// per src/scene/pbr.rs lobe_weights / under_layer_scale).

inline void pat_recompute_derived(thread PtMaterial& m) {
    float lum_f0 = 0.2126f * m.spec_f0[0] + 0.7152f * m.spec_f0[1]
        + 0.0722f * m.spec_f0[2];
    float coat_take = 0.04f * m.coat_gain;
    m.under_scale = clamp((1.0f - lum_f0) * (1.0f - coat_take), 0.0f, 1.0f);
    float lum_d = 0.2126f * m.diffuse_color[0] + 0.7152f * m.diffuse_color[1]
        + 0.0722f * m.diffuse_color[2];
    float lum_fz = 0.2126f * m.fuzz_color[0] + 0.7152f * m.fuzz_color[1]
        + 0.0722f * m.fuzz_color[2];
    float wd = m.diffuse_gain * lum_d;
    float ws = sqrt(lum_f0);
    float wc = m.coat_gain * 0.2f;
    float wf = m.fuzz_gain * lum_fz * 0.3f;
    float wg = m.glass_gain;
    float total = wd + ws + wc + wf + wg;
    if (total <= 1e-9f) {
        m.weights[0] = 0.0f; m.weights[1] = 0.0f; m.weights[2] = 0.0f;
        m.weights[3] = 0.0f; m.weights[4] = 0.0f;
    } else {
        m.weights[0] = wd / total;
        m.weights[1] = ws / total;
        m.weights[2] = wc / total;
        m.weights[3] = wf / total;
        m.weights[4] = wg / total;
    }
}
