//! Participating media (roadmap Phase 10): absorption + single/multiple
//! scattering with the Henyey-Greenstein phase function. Homogeneous
//! media sample distances analytically; heterogeneous media (procedural
//! fBm density for clouds) use delta tracking for distance sampling and
//! ratio tracking for transmittance, both against a conservative
//! majorant.
//!
//! Media attach to geometry as `Interior` (the hull's inside) or globally
//! as `Atmosphere`. A hull whose material has no lobes and no emission is
//! invisible: rays pass straight through and just toggle the medium.

use crate::geometry::displace::{fbm, DisplaceParams};
use crate::math::{Point3, Vec3};
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub enum DensityField {
    /// Puffy-cloud density: fBm remapped by coverage, faded at the hull
    /// boundary via a vertical falloff. Evaluated in world space.
    Fbm {
        params: DisplaceParams,
        /// Density threshold: higher coverage = fuller volume.
        coverage: f64,
        /// Sharpness of the density ramp above the threshold.
        sharpness: f64,
    },
}

#[derive(Debug, Clone)]
pub struct Medium {
    pub sigma_a: Vec3,
    pub sigma_s: Vec3,
    /// Henyey-Greenstein anisotropy in (-1, 1).
    pub g: f64,
    /// None = homogeneous; Some = density multiplier field in [0, 1].
    pub density: Option<DensityField>,
    /// Uniform emission (glowing media); rarely used, default zero.
    pub emission: Vec3,
    /// Extent of the medium along any ray (atmospheres are slabs, not
    /// infinite — an unbounded homogeneous fog absorbs the whole sky).
    pub max_distance: f64,
}

impl Medium {
    pub fn sigma_t(&self) -> Vec3 {
        self.sigma_a + self.sigma_s
    }

    /// Channel-max extinction — the delta-tracking majorant (density
    /// fields are clamped to [0,1] so sigma_t is its own bound).
    pub fn majorant(&self) -> f64 {
        let st = self.sigma_t();
        st.x.max(st.y).max(st.z)
    }

    /// Density multiplier at a world point.
    pub fn density_at(&self, p: &Point3) -> f64 {
        match &self.density {
            None => 1.0,
            Some(DensityField::Fbm { params, coverage, sharpness }) => {
                let n = fbm([p.x, p.y, p.z], params) * 0.5 + 0.5;
                ((n - (1.0 - coverage)) * sharpness).clamp(0.0, 1.0)
            }
        }
    }

    pub fn is_homogeneous(&self) -> bool {
        self.density.is_none()
    }
}

/// Henyey-Greenstein phase function value (also its own solid-angle pdf).
pub fn hg_phase(g: f64, cos_theta: f64) -> f64 {
    let denom = 1.0 + g * g + 2.0 * g * cos_theta;
    (1.0 - g * g) / (4.0 * PI * denom * denom.max(1e-9).sqrt())
}

/// Sample the HG phase function around `wo` (pointing away from the
/// collision, toward the previous vertex). Returns (wi, pdf).
pub fn hg_sample(g: f64, wo: &Vec3, u1: f64, u2: f64) -> (Vec3, f64) {
    let cos_theta = if g.abs() < 1e-3 {
        1.0 - 2.0 * u1
    } else {
        let sq = (1.0 - g * g) / (1.0 + g - 2.0 * g * u1);
        -(1.0 + g * g - sq * sq) / (2.0 * g)
    };
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = 2.0 * PI * u2;
    // Frame around -wo (the propagation direction).
    let w = -*wo;
    let a = if w.x.abs() > 0.9 {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let t = w.cross(&a).normalize();
    let b = w.cross(&t);
    let wi = (t * (sin_theta * phi.cos()) + b * (sin_theta * phi.sin()) + w * cos_theta)
        .normalize();
    (wi, hg_phase(g, wo.dot(&wi) * -1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hg_normalizes() {
        // ∫ phase dω = 1 for several g.
        for g in [-0.6, 0.0, 0.3, 0.8] {
            let n = 100_000;
            let mut sum = 0.0;
            let mut state = 0x12345678u64;
            let mut rand = || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state >> 33) as f64 / (1u64 << 31) as f64
            };
            for _ in 0..n {
                let z: f64 = 1.0 - 2.0 * rand();
                sum += hg_phase(g, z) * 2.0 * PI * 2.0; // uniform in cos: pdf = 1/2 over [-1,1]
            }
            let mean = sum / n as f64;
            assert!((mean - 1.0).abs() < 0.02, "g={g}: {mean}");
        }
    }

    #[test]
    fn hg_sample_matches_pdf() {
        // E[1/pdf * phase] over sampled directions = 1 (self-importance).
        let wo = Vec3::new(0.0, 0.0, 1.0);
        let mut state = 99u64;
        let mut rand = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as f64 / (1u64 << 31) as f64
        };
        for g in [0.0, 0.5, -0.4] {
            let n = 50_000;
            let mut sum = 0.0;
            for _ in 0..n {
                let (wi, pdf) = hg_sample(g, &wo, rand(), rand());
                let val = hg_phase(g, -wo.dot(&wi));
                sum += val / pdf.max(1e-12);
            }
            let mean = sum / n as f64;
            assert!((mean - 1.0).abs() < 0.01, "g={g}: {mean}");
        }
    }

    #[test]
    fn fbm_density_bounded() {
        let m = Medium {
            sigma_a: Vec3::new(0.1, 0.1, 0.1),
            sigma_s: Vec3::new(1.0, 1.0, 1.0),
            g: 0.4,
            density: Some(DensityField::Fbm {
                params: DisplaceParams { frequency: 0.5, octaves: 4, ..Default::default() },
                coverage: 0.55,
                sharpness: 4.0,
            }),
            emission: Vec3::zero(),
            max_distance: f64::INFINITY,
        };
        for i in 0..500 {
            let p = Point3::new(i as f64 * 0.37, (i % 17) as f64 * 0.9, (i % 5) as f64);
            let d = m.density_at(&p);
            assert!((0.0..=1.0).contains(&d), "density {d}");
        }
        assert!(m.majorant() > 1.0);
    }
}
