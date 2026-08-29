// Whitted kernel: direct lighting + hard shadows + mirror bounces.
// Compiled after isect_common.metal (see renderer.rs).

struct Uniforms {
    uint  width;
    uint  height;
    uint  samples_x;
    uint  samples_y;
    uint  object_count;
    uint  light_count;
    uint  max_depth;
    uint  y_offset;           // first image row of this dispatch band
    float background[3];
    float eye[3];
    float forward[3];
    float right[3];
    float up[3];
    float half_width;
    float half_height;
};

struct Material {
    float r, g, b;
    float ka, kd, ks;
    float shininess;          // precomputed shininess_for()
    float reflectivity;       // precomputed Material::reflectivity()
    uint  is_metal;
};

struct Light {
    uint  kind;               // 0 point (x,y,z = position), 1 distant (x,y,z = direction)
    float x, y, z;
    float intensity;
    float r, g, b;
};

struct SceneHit { bool hit; float t; float3 p; float3 n; uint obj; };

inline SceneHit isect_scene(device const Object* objects, uint count,
                            float3 wo, float3 wd) {
    SceneHit best;
    best.hit = false; best.t = INFINITY;
    best.p = float3(0.0f); best.n = float3(0.0f); best.obj = 0u;
    for (uint i = 0u; i < count; i++) {
        Hit h = isect_object(objects[i], wo, wd);
        if (h.valid && h.t < best.t) {           // strict <, Scene::intersect rule
            best.hit = true; best.t = h.t; best.p = h.p; best.n = h.n; best.obj = i;
        }
    }
    return best;
}

// Scene::is_occluded: origin nudged along the normal; a blocker must be
// nearer than max_t - SHADOW_EPS (INFINITY for distant lights — fast math
// is off, so the arithmetic is IEEE).
inline bool occluded(device const Object* objects, uint count,
                     float3 p, float3 n, float3 ldir, float max_t) {
    float3 origin = p + n * SHADOW_EPS;
    float limit = max_t - SHADOW_EPS;
    for (uint i = 0u; i < count; i++) {
        Hit h = isect_object(objects[i], origin, ldir);
        if (h.valid && h.t < limit) return true;
    }
    return false;
}

// ---- shading (src/shading/mod.rs, exact) ----

inline float3 shade_hit(constant Uniforms& u,
                        device const Object* objects,
                        device const Material* materials,
                        device const Light* lights,
                        SceneHit h) {
    Material m = materials[h.obj];               // per-object material table
    float3 mcol = float3(m.r, m.g, m.b);
    float3 color = mcol * m.ka;                  // ambient
    float3 eye = float3(u.eye[0], u.eye[1], u.eye[2]);
    float3 view = normalize_cpu(eye - h.p);      // always from the camera eye

    if (u.light_count == 0u) {
        // Headlight shading for depth perception.
        float ndv = max(dot(h.n, view), 0.0f);
        return color + mcol * (m.kd * ndv);
    }

    for (uint li = 0u; li < u.light_count; li++) {
        Light L = lights[li];
        float3 lvec = float3(L.x, L.y, L.z);
        float3 ldir;
        float max_t;
        if (L.kind == 0u) {                      // point light
            float3 to_light = lvec - h.p;
            max_t = length(to_light);
            ldir = normalize_cpu(to_light);
        } else {                                 // distant light
            ldir = -lvec;
            max_t = INFINITY;
        }
        float ndl = dot(h.n, ldir);
        if (ndl <= 0.0f) continue;
        if (occluded(objects, u.object_count, h.p, h.n, ldir, max_t)) continue;

        float3 lcol = float3(L.r, L.g, L.b);
        color += mcol * lcol * (m.kd * ndl * L.intensity);

        if (m.ks > 0.0f) {
            float3 half_v = normalize_cpu(ldir + view);
            float ndh = max(dot(h.n, half_v), 0.0f);
            // Metals tint specular; dielectrics reflect the light's color.
            float3 spec_col = (m.is_metal != 0u) ? mcol * lcol : lcol;
            color += spec_col * (m.ks * pow(ndh, m.shininess) * L.intensity);
        }
    }
    return color;
}

// ---- bounce loop (the throughput accumulation equivalent to the CPU's
//      trace_ray recursion: color*(1-r) + tint*r*(...)) ----

inline float3 trace(constant Uniforms& u,
                    device const Object* objects,
                    device const Material* materials,
                    device const Light* lights,
                    float3 origin, float3 dir) {
    float3 bg = float3(u.background[0], u.background[1], u.background[2]);
    float3 color = float3(0.0f);
    float3 throughput = float3(1.0f);

    for (uint depth = 0u; depth < u.max_depth; depth++) {
        SceneHit h = isect_scene(objects, u.object_count, origin, dir);
        if (!h.hit) return color + throughput * bg;

        float3 local = shade_hit(u, objects, materials, lights, h);
        Material m = materials[h.obj];
        float refl = m.reflectivity;
        color += throughput * local * (1.0f - refl);
        if (refl <= 0.0f) return color;

        float3 tint = (m.is_metal != 0u) ? float3(m.r, m.g, m.b) : float3(1.0f);
        throughput *= tint * refl;

        dir = dir - h.n * (2.0f * dot(dir, h.n));   // Vec3::reflect
        origin = h.p + h.n * REFL_EPS;
    }
    // Depth limit reached: resolve to background (CPU recursion base case).
    return color + throughput * bg;
}

// ---- entry point: one thread per pixel ----

kernel void render_pixels(device const Object*   objects   [[buffer(0)]],
                          device const Material* materials [[buffer(1)]],
                          device const Light*    lights    [[buffer(2)]],
                          constant Uniforms&     u         [[buffer(3)]],
                          device float*          out_rgba  [[buffer(4)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint x = gid.x;
    uint y = gid.y + u.y_offset;
    if (x >= u.width || y >= u.height) return;

    float3 eye = float3(u.eye[0], u.eye[1], u.eye[2]);
    float3 fwd = float3(u.forward[0], u.forward[1], u.forward[2]);
    float3 rgt = float3(u.right[0], u.right[1], u.right[2]);
    float3 upv = float3(u.up[0], u.up[1], u.up[2]);

    float3 sum = float3(0.0f);
    for (uint sy = 0u; sy < u.samples_y; sy++) {
        for (uint sx = 0u; sx < u.samples_x; sx++) {
            // Stratified subpixels + Camera::generate_ray, exactly.
            float px = (float)x + ((float)sx + 0.5f) / (float)u.samples_x;
            float py = (float)y + ((float)sy + 0.5f) / (float)u.samples_y;
            float uu = (px / (float)u.width) * 2.0f - 1.0f;
            float vv = 1.0f - (py / (float)u.height) * 2.0f;
            float3 dir = fwd + rgt * (uu * u.half_width) + upv * (vv * u.half_height);
            sum += trace(u, objects, materials, lights, eye, dir);
        }
    }
    float3 c = sum / (float)(u.samples_x * u.samples_y);
    uint idx = (y * u.width + x) * 4u;
    out_rgba[idx + 0u] = c.x;
    out_rgba[idx + 1u] = c.y;
    out_rgba[idx + 2u] = c.z;
    out_rgba[idx + 3u] = 1.0f;
}

// ---- test-only probe: intersect objects[0] with a buffer of rays, for the
//      per-primitive parity tests in tests/metal_parity.rs ----

struct ProbeRay { float ox, oy, oz, dx, dy, dz; };
struct ProbeHit { float valid; float t; };

kernel void intersect_probe(device const Object*   objects   [[buffer(0)]],
                            device const ProbeRay* rays      [[buffer(1)]],
                            device ProbeHit*       out_hits  [[buffer(2)]],
                            constant uint&         ray_count [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= ray_count) return;
    ProbeRay r = rays[gid];
    Hit h = isect_object(objects[0], float3(r.ox, r.oy, r.oz), float3(r.dx, r.dy, r.dz));
    out_hits[gid].valid = h.valid ? 1.0f : 0.0f;
    out_hits[gid].t = h.t;
}
