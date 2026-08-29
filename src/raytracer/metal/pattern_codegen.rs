//! MSL codegen for the pattern node graph (roadmap Phase 6). The scene's
//! pattern DAG becomes generated Metal functions compiled with the PT
//! kernel (reusing the runtime-MSL muscle), and every texture the graph
//! references is packed — all mip levels, row-major RGB — into one flat
//! buffer with a per-mip offset table. Trilinear sampling on the GPU
//! mirrors the CPU tile cache's math exactly; only the storage differs
//! (resident buffer vs demand-paged tiles).

use crate::math::Vec3;
use crate::scene::Scene;
use crate::texture::cache::Wrap;
use crate::texture::global_cache;
use crate::texture::pattern::{PInput, PatternNode, TextureRef};
use std::collections::HashMap;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuTexMip {
    /// Float index (not byte) of this mip's first channel in the packed
    /// texture buffer.
    pub offset: u32,
    pub width: u32,
    pub height: u32,
    pub pad: u32,
}

const _: () = assert!(std::mem::size_of::<GpuTexMip>() == 16);

/// Where a texture's mips live in the packed table.
#[derive(Clone, Copy)]
struct TexEntry {
    mip_start: u32,
    mip_count: u32,
    base_w: u32,
    base_h: u32,
}

pub struct PatternGpu {
    pub tex_data: Vec<f32>,
    pub tex_mips: Vec<GpuTexMip>,
    /// Generated MSL: pat_node_* functions + apply_patterns.
    pub msl: String,
}

fn wrap_code(w: Wrap) -> u32 {
    match w {
        Wrap::Periodic => 0,
        Wrap::Clamp => 1,
        Wrap::Black => 2,
    }
}

fn f(v: f64) -> String {
    // Stable float literal: always a decimal point so MSL sees a float.
    let s = format!("{:?}", v as f32);
    if s.contains('.') || s.contains('e') { format!("{s}f") } else { format!("{s}.0f") }
}

fn f3(v: &Vec3) -> String {
    format!("float3({}, {}, {})", f(v.x), f(v.y), f(v.z))
}

/// Pack every texture referenced by the graph into the flat table.
fn build_texture_table(
    patterns: &[PatternNode],
) -> (Vec<f32>, Vec<GpuTexMip>, HashMap<u32, TexEntry>) {
    let mut data = Vec::new();
    let mut mips = Vec::new();
    let mut entries: HashMap<u32, TexEntry> = HashMap::new();
    let cache = global_cache();

    let add_tex = |id: u32, data: &mut Vec<f32>, mips: &mut Vec<GpuTexMip>,
                       entries: &mut HashMap<u32, TexEntry>| {
        if entries.contains_key(&id) {
            return;
        }
        let header = cache.header(id);
        let mip_start = mips.len() as u32;
        for level in 0..header.mips.len() {
            let (w, h, pixels) = cache.read_mip(id, level);
            mips.push(GpuTexMip {
                offset: data.len() as u32,
                width: w,
                height: h,
                pad: 0,
            });
            data.extend_from_slice(&pixels);
        }
        entries.insert(
            id,
            TexEntry {
                mip_start,
                mip_count: header.mips.len() as u32,
                base_w: header.width,
                base_h: header.height,
            },
        );
    };

    for node in patterns {
        let texref = match node {
            PatternNode::Texture { tex, .. } | PatternNode::Triplanar { tex, .. } => tex,
            _ => continue,
        };
        match texref {
            TextureRef::Single(id) => add_tex(*id, &mut data, &mut mips, &mut entries),
            TextureRef::Udim(tiles) => {
                for id in tiles.values() {
                    add_tex(*id, &mut data, &mut mips, &mut entries);
                }
            }
            TextureRef::Missing => {}
        }
    }
    (data, mips, entries)
}

/// A `tex_sample(...)` call with the texture's table location inlined.
/// `s_expr`/`t_expr` are in st space (t up); the image-space flip
/// (t = 1 - t) happens here, after any UDIM tile pick — matching the CPU's
/// sample_tex.
fn emit_tex_sample(
    entries: &HashMap<u32, TexEntry>,
    tex: &TextureRef,
    wrap: Wrap,
    s_expr: &str,
    t_expr: &str,
    fp_expr: &str,
) -> String {
    let one = |id: u32, s: &str, t: &str| -> String {
        let e = &entries[&id];
        format!(
            "tex_sample(td, tm, {}u, {}u, {}u, {}u, {s}, (1.0f - ({t})), {fp_expr}, {}u)",
            e.mip_start, e.mip_count, e.base_w, e.base_h, wrap_code(wrap)
        )
    };
    match tex {
        TextureRef::Single(id) => one(*id, s_expr, t_expr),
        TextureRef::Missing => "float3(1.0f, 0.0f, 1.0f)".to_string(),
        TextureRef::Udim(tiles) => {
            // Tile pick mirrors TextureRef::resolve; a conditional chain
            // over the tiles that exist.
            let mut out = format!(
                "({{ float _us = {s_expr}; float _ut = {t_expr}; \
                 float _sc = clamp(floor(_us), 0.0f, 9.0f); \
                 float _tc = max(floor(_ut), 0.0f); \
                 int _tile = 1001 + (int)_sc + 10 * (int)_tc; \
                 float3 _r = float3(1.0f, 0.0f, 1.0f);"
            );
            let mut keys: Vec<_> = tiles.keys().copied().collect();
            keys.sort_unstable();
            for k in keys {
                out.push_str(&format!(
                    " if (_tile == {k}) _r = {};",
                    one(tiles[&k], "(_us - _sc)", "(_ut - _tc)")
                ));
            }
            out.push_str(" _r; })");
            out
        }
    }
}

fn emit_input(input: &PInput) -> String {
    match input {
        PInput::Const(c) => f3(c),
        PInput::Node(i) => format!("pat_node_{i}(st, P, N, fp, td, tm)"),
    }
}

fn emit_node(entries: &HashMap<u32, TexEntry>, index: usize, node: &PatternNode) -> String {
    let body = match node {
        PatternNode::Texture { tex, wrap, scale } => {
            let s = format!("(st.x * {})", f(scale[0]));
            let t = format!("(st.y * {})", f(scale[1]));
            let fp = format!("(fp * {})", f(scale[0].abs().max(scale[1].abs())));
            format!("return {};", emit_tex_sample(entries, tex, *wrap, &s, &t, &fp))
        }
        PatternNode::Checker { color_a, color_b, scale } => format!(
            "float fs = st.x * {sx}; float ft = st.y * {sy};\n    \
             int cell = ((int)floor(fs) + (int)floor(ft)) & 1;\n    \
             float3 sharp = cell == 0 ? {ca} : {cb};\n    \
             float3 mean = ({ca} + {cb}) * 0.5f;\n    \
             float w = min(fp * {smax}, 1.0f);\n    \
             return sharp * (1.0f - w) + mean * w;",
            sx = f(scale[0]),
            sy = f(scale[1]),
            ca = f3(color_a),
            cb = f3(color_b),
            smax = f(scale[0].abs().max(scale[1].abs())),
        ),
        PatternNode::Fractal { frequency, octaves, gain, lacunarity } => format!(
            "float v = clamp(pat_fbm(P, {}, {}u, {}, {}) * 0.5f + 0.5f, 0.0f, 1.0f);\n    \
             return float3(v, v, v);",
            f(*frequency),
            (*octaves).max(1),
            f(*gain),
            f(*lacunarity),
        ),
        PatternNode::Mix { color1, color2, mix } => format!(
            "float3 a = {};\n    float3 b = {};\n    \
             float m = clamp({}.x, 0.0f, 1.0f);\n    \
             return a * (1.0f - m) + b * m;",
            emit_input(color1),
            emit_input(color2),
            emit_input(mix),
        ),
        PatternNode::ColorCorrect { input, gain, offset, gamma, saturation } => format!(
            "float3 c = {};\n    float g = 1.0f / max({}, 1e-6f);\n    \
             float3 o = pow(max(c * {} + {}, 0.0f), float3(g));\n    \
             float lum = 0.2126f * o.x + 0.7152f * o.y + 0.0722f * o.z;\n    \
             return float3(lum) + (o - float3(lum)) * {};",
            emit_input(input),
            f(*gamma),
            f3(gain),
            f3(offset),
            f(*saturation),
        ),
        PatternNode::Ramp { positions, colors, use_t } => {
            let axis = if *use_t { "st.y" } else { "st.x" };
            let mut s = format!("float x = clamp({axis}, 0.0f, 1.0f);\n");
            if colors.is_empty() {
                s.push_str("    return float3(1.0f, 0.0f, 1.0f);");
            } else {
                s.push_str(&format!(
                    "    if (x <= {}) return {};\n",
                    f(positions[0]),
                    f3(&colors[0])
                ));
                for i in 1..positions.len().min(colors.len()) {
                    s.push_str(&format!(
                        "    if (x <= {p1}) {{ float a = (x - {p0}) / max({p1} - {p0}, 1e-9f); \
                         return {c0} * (1.0f - a) + {c1} * a; }}\n",
                        p0 = f(positions[i - 1]),
                        p1 = f(positions[i]),
                        c0 = f3(&colors[i - 1]),
                        c1 = f3(&colors[i]),
                    ));
                }
                s.push_str(&format!("    return {};", f3(colors.last().unwrap())));
            }
            s
        }
        PatternNode::Triplanar { tex, wrap, frequency } => {
            let freq = f(*frequency);
            format!(
                "float3 an = fabs(N);\n    \
                 float total = max(an.x + an.y + an.z, 1e-9f);\n    \
                 float3 w = an / total;\n    \
                 float3 cx = {};\n    float3 cy = {};\n    float3 cz = {};\n    \
                 return cx * w.x + cy * w.y + cz * w.z;",
                emit_tex_sample(entries, tex, *wrap, &format!("(P.y * {freq})"), &format!("(P.z * {freq})"), "fp"),
                emit_tex_sample(entries, tex, *wrap, &format!("(P.z * {freq})"), &format!("(P.x * {freq})"), "fp"),
                emit_tex_sample(entries, tex, *wrap, &format!("(P.x * {freq})"), &format!("(P.y * {freq})"), "fp"),
            )
        }
    };
    format!(
        "static float3 pat_node_{index}(float2 st, float3 P, float3 N, float fp,\n\
         \x20                          device const float* td, device const TexMipG* tm) {{\n    {body}\n}}\n"
    )
}

/// Build the packed texture table and the generated MSL for a scene.
pub fn build(scene: &Scene) -> PatternGpu {
    let (tex_data, tex_mips, entries) = build_texture_table(&scene.patterns);

    let mut msl = String::new();
    for (i, node) in scene.patterns.iter().enumerate() {
        msl.push_str(&emit_node(&entries, i, node));
    }

    // apply_patterns: per-material bound-field overrides + recomputation
    // of the derived quantities the kernel consumes (alphas store
    // roughness^2; weights/under_scale depend on colors and gains).
    msl.push_str(
        "static void apply_patterns(uint mat_id, thread PtMaterial& m,\n\
         \x20                         float2 st, float3 P, float3 N, float fp,\n\
         \x20                         device const float* td, device const TexMipG* tm) {\n",
    );
    let bound: Vec<(usize, &crate::scene::Material)> = scene
        .materials
        .iter()
        .enumerate()
        .filter(|(_, m)| !m.pattern_bindings.is_empty())
        .collect();
    if bound.is_empty() {
        msl.push_str("    (void)mat_id; (void)m; (void)st; (void)P; (void)N; (void)fp; (void)td; (void)tm;\n}\n");
    } else {
        msl.push_str("    switch (mat_id) {\n");
        for (id, material) in bound {
            msl.push_str(&format!("    case {id}u: {{\n"));
            for (field, node) in &material.pattern_bindings {
                let v = format!("pat_node_{node}(st, P, N, fp, td, tm)");
                use crate::texture::pattern::BoundField as B;
                let assign = match field {
                    B::DiffuseColor => format!("float3 v = {v}; m.diffuse_color[0] = v.x; m.diffuse_color[1] = v.y; m.diffuse_color[2] = v.z;"),
                    B::DiffuseGain => format!("m.diffuse_gain = {v}.x;"),
                    B::DiffuseRoughness => format!("m.diffuse_sigma = {v}.x;"),
                    B::SpecularFaceColor => format!("float3 v = {v}; m.spec_f0[0] = v.x; m.spec_f0[1] = v.y; m.spec_f0[2] = v.z;"),
                    B::SpecularEdgeColor => format!("float3 v = {v}; m.spec_f90[0] = v.x; m.spec_f90[1] = v.y; m.spec_f90[2] = v.z;"),
                    B::SpecularRoughness => format!("float r = clamp({v}.x, 0.005f, 1.0f); m.spec_alpha = clamp(r * r, 2.5e-5f, 1.0f);"),
                    B::ClearcoatGain => format!("m.coat_gain = {v}.x;"),
                    B::FuzzGain => format!("m.fuzz_gain = {v}.x;"),
                    B::FuzzColor => format!("float3 v = {v}; m.fuzz_color[0] = v.x; m.fuzz_color[1] = v.y; m.fuzz_color[2] = v.z;"),
                    B::GlassGain => format!("m.glass_gain = {v}.x;"),
                    B::GlassRoughness => format!("float r = clamp({v}.x, 0.005f, 1.0f); m.glass_alpha = clamp(r * r, 2.5e-5f, 1.0f);"),
                    B::RefractionColor => format!("float3 v = {v}; m.refr_color[0] = v.x; m.refr_color[1] = v.y; m.refr_color[2] = v.z;"),
                    B::Glow => format!("float3 v = {v}; m.emission[0] = v.x; m.emission[1] = v.y; m.emission[2] = v.z;"),
                    B::Presence => format!("m.presence = clamp({v}.x, 0.0f, 1.0f);"),
                };
                msl.push_str(&format!("        {{ {assign} }}\n"));
            }
            msl.push_str("        pat_recompute_derived(m);\n        break;\n    }\n");
        }
        msl.push_str("    default: break;\n    }\n}\n");
    }

    PatternGpu { tex_data, tex_mips, msl }
}
