//! PxrSurface-lite material parameters (roadmap Phase 4): a lobe stack
//! with PRMan-compatible parameter names, consumed by the path tracer's
//! bxdf module. The legacy matte/plastic/metal surfaces map onto it so old
//! scenes keep rendering (the Whitted integrator ignores this and keeps
//! its Phong terms).

use crate::math::Vec3;
use crate::parser::ParamList;

#[derive(Debug, Clone)]
pub struct PbrParams {
    pub diffuse_gain: f64,
    pub diffuse_color: Vec3,
    /// Oren-Nayar sigma in [0,1] (0 = Lambert).
    pub diffuse_roughness: f64,

    /// Schlick F0; black disables the lobe (unless specular_ior > 1).
    pub specular_face_color: Vec3,
    /// Schlick F90 (grazing tint).
    pub specular_edge_color: Vec3,
    pub specular_roughness: f64,
    /// When face color is black, F0 derives from this dielectric IOR.
    pub specular_ior: f64,

    pub clearcoat_gain: f64,
    pub clearcoat_roughness: f64,

    pub fuzz_gain: f64,
    pub fuzz_color: Vec3,

    /// glassRefractionGain: rough dielectric with true refraction.
    pub glass_gain: f64,
    pub glass_ior: f64,
    pub glass_roughness: f64,
    pub refraction_color: Vec3,

    pub glow: Vec3,
    /// Stochastic cutout alpha (1 = solid).
    pub presence: f64,
}

impl Default for PbrParams {
    fn default() -> Self {
        Self {
            diffuse_gain: 1.0,
            diffuse_color: Vec3::new(0.5, 0.5, 0.5),
            diffuse_roughness: 0.0,
            specular_face_color: Vec3::zero(),
            specular_edge_color: Vec3::one(),
            specular_roughness: 0.2,
            specular_ior: 1.5,
            clearcoat_gain: 0.0,
            clearcoat_roughness: 0.1,
            fuzz_gain: 0.0,
            fuzz_color: Vec3::one(),
            glass_gain: 0.0,
            glass_ior: 1.5,
            glass_roughness: 0.05,
            refraction_color: Vec3::one(),
            glow: Vec3::zero(),
            presence: 1.0,
        }
    }
}

fn lum(c: &Vec3) -> f64 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

impl PbrParams {
    /// Effective specular F0: explicit face color, else derived from the
    /// dielectric IOR.
    pub fn specular_f0(&self) -> Vec3 {
        if lum(&self.specular_face_color) > 1e-6 {
            self.specular_face_color
        } else if self.specular_ior > 1.0 {
            let r = (self.specular_ior - 1.0) / (self.specular_ior + 1.0);
            Vec3::one() * (r * r)
        } else {
            Vec3::zero()
        }
    }

    /// Whether the primary specular lobe is active at all. Legacy matte
    /// maps to a params set with specular disabled via specular_ior = 1.
    pub fn has_specular(&self) -> bool {
        lum(&self.specular_f0()) > 1e-6
    }

    /// Legacy surface mappings ------------------------------------------

    pub fn from_matte(color: Vec3) -> Self {
        Self {
            diffuse_gain: 1.0,
            diffuse_color: color,
            specular_ior: 1.0, // no specular
            ..Default::default()
        }
    }

    pub fn from_plastic(color: Vec3, roughness: f64) -> Self {
        Self {
            diffuse_gain: 0.9,
            diffuse_color: color,
            specular_ior: 1.5, // F0 = 0.04
            specular_roughness: roughness.max(0.02),
            ..Default::default()
        }
    }

    pub fn from_metal(color: Vec3, roughness: f64) -> Self {
        Self {
            diffuse_gain: 0.0,
            specular_face_color: color,
            specular_edge_color: Vec3::one(),
            specular_roughness: roughness.max(0.02),
            ..Default::default()
        }
    }

    /// Parse `Bxdf "PxrSurface" ...` parameters (PRMan-compatible names).
    pub fn from_bxdf_params(params: &ParamList<'_>) -> Self {
        let mut p = Self {
            diffuse_gain: 1.0,
            diffuse_color: Vec3::new(0.18, 0.18, 0.18),
            specular_ior: 1.5,
            specular_roughness: 0.2,
            ..Default::default()
        };
        let color = |name: &str, dst: &mut Vec3| {
            if let Some(v) = params.get_numbers(name) {
                if v.len() >= 3 {
                    *dst = Vec3::new(v[0], v[1], v[2]);
                }
            }
        };
        let scalar = |name: &str, dst: &mut f64| {
            if let Some(v) = params.get_number(name) {
                *dst = v;
            }
        };
        scalar("diffuseGain", &mut p.diffuse_gain);
        color("diffuseColor", &mut p.diffuse_color);
        scalar("diffuseRoughness", &mut p.diffuse_roughness);
        color("specularFaceColor", &mut p.specular_face_color);
        color("specularEdgeColor", &mut p.specular_edge_color);
        scalar("specularRoughness", &mut p.specular_roughness);
        scalar("specularIor", &mut p.specular_ior);
        scalar("clearcoatGain", &mut p.clearcoat_gain);
        scalar("clearcoatRoughness", &mut p.clearcoat_roughness);
        scalar("fuzzGain", &mut p.fuzz_gain);
        color("fuzzColor", &mut p.fuzz_color);
        scalar("glassRefractionGain", &mut p.glass_gain);
        scalar("glassIor", &mut p.glass_ior);
        scalar("glassRoughness", &mut p.glass_roughness);
        color("refractionColor", &mut p.refraction_color);
        let mut glow_gain = 0.0;
        let mut glow_color = Vec3::one();
        scalar("glowGain", &mut glow_gain);
        color("glowColor", &mut glow_color);
        p.glow = glow_color * glow_gain;
        scalar("presence", &mut p.presence);
        p.specular_roughness = p.specular_roughness.clamp(0.005, 1.0);
        p.glass_roughness = p.glass_roughness.clamp(0.005, 1.0);
        p.clearcoat_roughness = p.clearcoat_roughness.clamp(0.005, 1.0);
        p
    }

    /// Lobe-selection weights (diffuse, specular, clearcoat, fuzz, glass),
    /// normalized; None when the material reflects nothing.
    pub fn lobe_weights(&self) -> Option<[f64; 5]> {
        let wd = self.diffuse_gain * lum(&self.diffuse_color);
        let ws = lum(&self.specular_f0()).sqrt(); // grazing boost: F90 matters
        let wc = self.clearcoat_gain * 0.2;
        let wf = self.fuzz_gain * lum(&self.fuzz_color) * 0.3;
        let wg = self.glass_gain;
        let total = wd + ws + wc + wf + wg;
        if total <= 1e-9 {
            return None;
        }
        Some([wd / total, ws / total, wc / total, wf / total, wg / total])
    }
}
