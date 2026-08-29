//! The native pattern node graph (roadmap Phase 6): data-only nodes built
//! from `Pattern` RIB requests, evaluated per shading point on the CPU
//! (the Metal backend consumes the same nodes through MSL codegen).
//! Connections use PRMan's `"reference color diffuseColor"
//! ["nodeHandle:resultRGB"]` convention; every node outputs RGB, and
//! scalar consumers read channel 0.

use super::cache::{global as global_cache, TexId, Wrap};
use crate::geometry::displace::{fbm, DisplaceParams};
use crate::math::Vec3;
use std::collections::HashMap;

/// A texture file binding: one flat file or a UDIM tile set.
#[derive(Debug, Clone)]
pub enum TextureRef {
    Single(TexId),
    /// UDIM tiles keyed by tile number (1001 + s_cell + 10 * t_cell).
    Udim(HashMap<u16, TexId>),
    /// Open failed: sample as magenta so the error is visible in renders.
    Missing,
}

impl TextureRef {
    /// Resolve st to (texture, local st) — the UDIM tile pick.
    fn resolve(&self, s: f64, t: f64) -> Option<(TexId, f64, f64)> {
        match self {
            TextureRef::Single(id) => Some((*id, s, t)),
            TextureRef::Udim(tiles) => {
                let sc = s.floor().clamp(0.0, 9.0);
                let tc = t.floor().max(0.0);
                let tile = 1001 + sc as u16 + 10 * tc as u16;
                tiles.get(&tile).map(|id| (*id, s - sc, t - tc))
            }
            TextureRef::Missing => None,
        }
    }
}

/// Node input: a constant or another node's RGB output.
#[derive(Debug, Clone, Copy)]
pub enum PInput {
    Const(Vec3),
    Node(u32),
}

#[derive(Debug, Clone)]
pub enum PatternNode {
    /// PxrTexture: st-mapped file texture. `scale` multiplies st before
    /// lookup; `invert_t` matches texture-space vs st-space conventions.
    Texture { tex: TextureRef, wrap: Wrap, scale: [f64; 2] },
    /// PxrChecker (with sScale/tScale extensions).
    Checker { color_a: Vec3, color_b: Vec3, scale: [f64; 2] },
    /// PxrFractal: fBm noise over P (world space) remapped to [0,1].
    Fractal { frequency: f64, octaves: u32, gain: f64, lacunarity: f64 },
    /// PxrMix: lerp(color1, color2, mix.r).
    Mix { color1: PInput, color2: PInput, mix: PInput },
    /// PxrColorCorrect subset: out = ((in * gain + offset) ^ (1/gamma)),
    /// then saturation about luminance.
    ColorCorrect { input: PInput, gain: Vec3, offset: Vec3, gamma: f64, saturation: f64 },
    /// PxrRamp subset: piecewise-linear color ramp over s or t.
    Ramp { positions: Vec<f64>, colors: Vec<Vec3>, use_t: bool },
    /// Triplanar projection of a file texture (blend by |N| axis weights).
    Triplanar { tex: TextureRef, wrap: Wrap, frequency: f64 },
}

/// Everything a pattern can depend on at one shading point.
#[derive(Debug, Clone, Copy)]
pub struct ShadeCtx {
    pub st: [f64; 2],
    /// World-space position.
    pub p: [f64; 3],
    /// Shading normal (world).
    pub n: [f64; 3],
    /// Pixel footprint diameter in st units (ray cone at the hit).
    pub footprint: f64,
}

const MISSING_COLOR: Vec3 = Vec3 { x: 1.0, y: 0.0, z: 1.0 };

fn eval_input(nodes: &[PatternNode], input: &PInput, ctx: &ShadeCtx, depth: u32) -> Vec3 {
    match input {
        PInput::Const(c) => *c,
        PInput::Node(i) => eval_at(nodes, *i, ctx, depth),
    }
}

fn sample_tex(tex: &TextureRef, wrap: Wrap, s: f64, t: f64, footprint: f64) -> Vec3 {
    match tex.resolve(s, t) {
        Some((id, s, t)) => {
            // st has t up; images store rows top-down.
            let c = global_cache().sample(id, s, 1.0 - t, footprint, wrap);
            Vec3::new(c[0] as f64, c[1] as f64, c[2] as f64)
        }
        None => MISSING_COLOR,
    }
}

fn eval_at(nodes: &[PatternNode], index: u32, ctx: &ShadeCtx, depth: u32) -> Vec3 {
    if depth > 16 {
        return MISSING_COLOR; // cycle guard; builder-produced graphs are DAGs
    }
    let Some(node) = nodes.get(index as usize) else {
        return MISSING_COLOR;
    };
    match node {
        PatternNode::Texture { tex, wrap, scale } => {
            let s = ctx.st[0] * scale[0];
            let t = ctx.st[1] * scale[1];
            sample_tex(tex, *wrap, s, t, ctx.footprint * scale[0].abs().max(scale[1].abs()))
        }
        PatternNode::Checker { color_a, color_b, scale } => {
            // Analytically filtered checker: blend toward the mean as the
            // footprint approaches the cell size (kills grazing shimmer).
            let fs = ctx.st[0] * scale[0];
            let ft = ctx.st[1] * scale[1];
            let cell = ((fs.floor() + ft.floor()) as i64).rem_euclid(2);
            let sharp = if cell == 0 { *color_a } else { *color_b };
            let mean = (*color_a + *color_b) * 0.5;
            let w = (ctx.footprint * scale[0].abs().max(scale[1].abs())).min(1.0);
            sharp * (1.0 - w) + mean * w
        }
        PatternNode::Fractal { frequency, octaves, gain, lacunarity } => {
            let params = DisplaceParams {
                amplitude: 1.0,
                frequency: *frequency,
                octaves: *octaves,
                gain: *gain,
                lacunarity: *lacunarity,
                offset: [0.0; 3],
            };
            let v = (fbm(ctx.p, &params) * 0.5 + 0.5).clamp(0.0, 1.0);
            Vec3::new(v, v, v)
        }
        PatternNode::Mix { color1, color2, mix } => {
            let a = eval_input(nodes, color1, ctx, depth + 1);
            let b = eval_input(nodes, color2, ctx, depth + 1);
            let m = eval_input(nodes, mix, ctx, depth + 1).x.clamp(0.0, 1.0);
            a * (1.0 - m) + b * m
        }
        PatternNode::ColorCorrect { input, gain, offset, gamma, saturation } => {
            let c = eval_input(nodes, input, ctx, depth + 1);
            let g = 1.0 / gamma.max(1e-6);
            let mut out = Vec3::new(
                (c.x * gain.x + offset.x).max(0.0).powf(g),
                (c.y * gain.y + offset.y).max(0.0).powf(g),
                (c.z * gain.z + offset.z).max(0.0).powf(g),
            );
            if (*saturation - 1.0).abs() > 1e-9 {
                let lum = 0.2126 * out.x + 0.7152 * out.y + 0.0722 * out.z;
                let grey = Vec3::new(lum, lum, lum);
                out = grey + (out - grey) * *saturation;
            }
            out
        }
        PatternNode::Ramp { positions, colors, use_t } => {
            let x = if *use_t { ctx.st[1] } else { ctx.st[0] }.clamp(0.0, 1.0);
            if colors.is_empty() {
                return MISSING_COLOR;
            }
            if x <= positions[0] {
                return colors[0];
            }
            for i in 1..positions.len() {
                if x <= positions[i] {
                    let span = (positions[i] - positions[i - 1]).max(1e-9);
                    let a = (x - positions[i - 1]) / span;
                    return colors[i - 1] * (1.0 - a) + colors[i] * a;
                }
            }
            *colors.last().unwrap()
        }
        PatternNode::Triplanar { tex, wrap, frequency } => {
            let n = Vec3::new(ctx.n[0].abs(), ctx.n[1].abs(), ctx.n[2].abs());
            let total = (n.x + n.y + n.z).max(1e-9);
            let w = Vec3::new(n.x / total, n.y / total, n.z / total);
            let f = *frequency;
            // Footprint in projected-uv units: world footprint * frequency.
            // ctx.footprint is st-based; approximate with the same value.
            let fp = ctx.footprint;
            let cx = sample_tex(tex, *wrap, ctx.p[1] * f, ctx.p[2] * f, fp);
            let cy = sample_tex(tex, *wrap, ctx.p[2] * f, ctx.p[0] * f, fp);
            let cz = sample_tex(tex, *wrap, ctx.p[0] * f, ctx.p[1] * f, fp);
            cx * w.x + cy * w.y + cz * w.z
        }
    }
}

/// Evaluate node `index` of the graph at a shading point.
pub fn eval(nodes: &[PatternNode], index: u32, ctx: &ShadeCtx) -> Vec3 {
    eval_at(nodes, index, ctx, 0)
}

/// PbrParams fields a pattern output can drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundField {
    DiffuseColor,
    DiffuseGain,
    DiffuseRoughness,
    SpecularFaceColor,
    SpecularEdgeColor,
    SpecularRoughness,
    ClearcoatGain,
    FuzzGain,
    FuzzColor,
    GlassGain,
    GlassRoughness,
    RefractionColor,
    Glow,
    Presence,
}

impl BoundField {
    pub fn from_param(name: &str) -> Option<Self> {
        Some(match name {
            "diffuseColor" => Self::DiffuseColor,
            "diffuseGain" => Self::DiffuseGain,
            "diffuseRoughness" => Self::DiffuseRoughness,
            "specularFaceColor" => Self::SpecularFaceColor,
            "specularEdgeColor" => Self::SpecularEdgeColor,
            "specularRoughness" => Self::SpecularRoughness,
            "clearcoatGain" => Self::ClearcoatGain,
            "fuzzGain" => Self::FuzzGain,
            "fuzzColor" => Self::FuzzColor,
            "glassRefractionGain" => Self::GlassGain,
            "glassRoughness" => Self::GlassRoughness,
            "refractionColor" => Self::RefractionColor,
            "glowColor" => Self::Glow,
            "presence" => Self::Presence,
            _ => return None,
        })
    }

    /// Write a pattern's RGB output into the field (scalars take .x).
    pub fn apply(&self, pbr: &mut crate::scene::PbrParams, v: Vec3) {
        match self {
            Self::DiffuseColor => pbr.diffuse_color = v,
            Self::DiffuseGain => pbr.diffuse_gain = v.x,
            Self::DiffuseRoughness => pbr.diffuse_roughness = v.x,
            Self::SpecularFaceColor => pbr.specular_face_color = v,
            Self::SpecularEdgeColor => pbr.specular_edge_color = v,
            Self::SpecularRoughness => pbr.specular_roughness = v.x.clamp(0.005, 1.0),
            Self::ClearcoatGain => pbr.clearcoat_gain = v.x,
            Self::FuzzGain => pbr.fuzz_gain = v.x,
            Self::FuzzColor => pbr.fuzz_color = v,
            Self::GlassGain => pbr.glass_gain = v.x,
            Self::GlassRoughness => pbr.glass_roughness = v.x.clamp(0.005, 1.0),
            Self::RefractionColor => pbr.refraction_color = v,
            Self::Glow => pbr.glow = v,
            Self::Presence => pbr.presence = v.x.clamp(0.0, 1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(st: [f64; 2]) -> ShadeCtx {
        ShadeCtx { st, p: [0.0; 3], n: [0.0, 0.0, 1.0], footprint: 0.0 }
    }

    #[test]
    fn checker_mix_ramp_chain() {
        let nodes = vec![
            PatternNode::Checker {
                color_a: Vec3::new(1.0, 0.0, 0.0),
                color_b: Vec3::new(0.0, 1.0, 0.0),
                scale: [2.0, 2.0],
            },
            PatternNode::Mix {
                color1: PInput::Node(0),
                color2: PInput::Const(Vec3::new(0.0, 0.0, 1.0)),
                mix: PInput::Const(Vec3::new(0.5, 0.0, 0.0)),
            },
            PatternNode::Ramp {
                positions: vec![0.0, 1.0],
                colors: vec![Vec3::zero(), Vec3::one()],
                use_t: false,
            },
        ];
        // (0.25, 0.25) is cell (0,0): color_a red.
        let c = eval(&nodes, 0, &ctx([0.25, 0.25]));
        assert!((c.x - 1.0).abs() < 1e-9 && c.y.abs() < 1e-9);
        // (0.75, 0.25) is cell (1,0): green.
        let c = eval(&nodes, 0, &ctx([0.75, 0.25]));
        assert!((c.y - 1.0).abs() < 1e-9);
        // Mix: half red + half blue.
        let m = eval(&nodes, 1, &ctx([0.25, 0.25]));
        assert!((m.x - 0.5).abs() < 1e-9 && (m.z - 0.5).abs() < 1e-9);
        // Ramp mid-point.
        let r = eval(&nodes, 2, &ctx([0.5, 0.0]));
        assert!((r.x - 0.5).abs() < 1e-9);
        // Wide footprint blends the checker toward its mean.
        let wide = ShadeCtx { footprint: 0.5, ..ctx([0.25, 0.25]) };
        let f = eval(&nodes, 0, &wide);
        assert!(f.x < 1.0 && f.y > 0.0, "{f:?}");
    }

    #[test]
    fn colorcorrect_gamma_and_saturation() {
        let nodes = vec![PatternNode::ColorCorrect {
            input: PInput::Const(Vec3::new(0.25, 0.25, 0.25)),
            gain: Vec3::one(),
            offset: Vec3::zero(),
            gamma: 2.0,
            saturation: 0.0,
        }];
        let c = eval(&nodes, 0, &ctx([0.0, 0.0]));
        // 0.25^(1/2) = 0.5, fully desaturated stays grey.
        assert!((c.x - 0.5).abs() < 1e-9 && (c.y - 0.5).abs() < 1e-9);
    }

    #[test]
    fn udim_tile_pick() {
        let mut tiles = HashMap::new();
        tiles.insert(1001u16, 7 as TexId);
        tiles.insert(1012u16, 9 as TexId);
        let t = TextureRef::Udim(tiles);
        let (id, s, _) = t.resolve(0.5, 0.5).unwrap();
        assert_eq!(id, 7);
        assert!((s - 0.5).abs() < 1e-12);
        // (1.3, 1.4) -> tile 1012, local (0.3, 0.4).
        let (id, s, tt) = t.resolve(1.3, 1.4).unwrap();
        assert_eq!(id, 9);
        assert!((s - 0.3).abs() < 1e-12 && (tt - 0.4).abs() < 1e-12);
        assert!(t.resolve(5.0, 0.0).is_none());
    }
}
