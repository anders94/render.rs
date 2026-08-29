//! Physically-based BSDF lobes over PbrParams (roadmap Phase 4): an
//! Oren-Nayar diffuse, GGX specular with VNDF sampling and height-correlated
//! Smith, a clearcoat, a fuzz/sheen term, and a rough dielectric (glass)
//! with true refraction (Walter/PBRT conventions, radiance transport).
//!
//! All lobe math runs in the local shading frame (z = shading normal, which
//! faces the viewer). `eta_rel` is the transmitted-side IOR over the
//! incident-side IOR (ior when entering, 1/ior when exiting), decided by
//! the integrator from the geometric normal.

use super::sampler::Pcg32;
use crate::math::Vec3;
use crate::scene::PbrParams;
use std::f64::consts::PI;

pub struct Frame {
    t: Vec3,
    b: Vec3,
    n: Vec3,
}

impl Frame {
    pub fn new(n: Vec3) -> Self {
        let t0 = if n.x.abs() > 0.9 {
            Vec3::new(0.0, 1.0, 0.0)
        } else {
            Vec3::new(1.0, 0.0, 0.0)
        };
        let t = n.cross(&t0).normalize();
        let b = n.cross(&t);
        Self { t, b, n }
    }
    pub fn to_local(&self, w: &Vec3) -> Vec3 {
        Vec3::new(w.dot(&self.t), w.dot(&self.b), w.dot(&self.n))
    }
    pub fn to_world(&self, w: &Vec3) -> Vec3 {
        self.t * w.x + self.b * w.y + self.n * w.z
    }
}

pub struct BsdfSample {
    /// Local-frame incoming direction (z < 0 for transmission).
    pub wi: Vec3,
    pub f: Vec3,
    pub pdf: f64,
    pub transmitted: bool,
}

#[allow(dead_code)]
fn lum(c: &Vec3) -> f64 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

fn schlick(f0: Vec3, f90: Vec3, cos: f64) -> Vec3 {
    let m = (1.0 - cos).clamp(0.0, 1.0).powi(5);
    f0 + (f90 - f0) * m
}

/// Exact dielectric Fresnel; eta = transmitted/incident IOR.
pub fn fresnel_dielectric(cos_i: f64, eta: f64) -> f64 {
    let cos_i = cos_i.clamp(0.0, 1.0);
    let sin2_t = (1.0 - cos_i * cos_i) / (eta * eta);
    if sin2_t >= 1.0 {
        return 1.0; // total internal reflection
    }
    let cos_t = (1.0 - sin2_t).sqrt();
    let r_par = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    let r_perp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    0.5 * (r_par * r_par + r_perp * r_perp)
}

// ---- GGX with height-correlated Smith + VNDF sampling -------------------

fn ggx_d(h: &Vec3, alpha: f64) -> f64 {
    if h.z <= 0.0 {
        return 0.0;
    }
    let a2 = alpha * alpha;
    let d = h.z * h.z * (a2 - 1.0) + 1.0;
    a2 / (PI * d * d)
}

fn ggx_lambda(w: &Vec3, alpha: f64) -> f64 {
    let cos2 = w.z * w.z;
    if cos2 <= 0.0 {
        return 0.0;
    }
    let tan2 = (1.0 - cos2).max(0.0) / cos2;
    ((1.0 + alpha * alpha * tan2).sqrt() - 1.0) * 0.5
}

fn ggx_g1(w: &Vec3, alpha: f64) -> f64 {
    1.0 / (1.0 + ggx_lambda(w, alpha))
}

fn ggx_g2(wo: &Vec3, wi: &Vec3, alpha: f64) -> f64 {
    1.0 / (1.0 + ggx_lambda(wo, alpha) + ggx_lambda(wi, alpha))
}

/// Heitz 2018 VNDF sampling; wo.z > 0.
fn ggx_sample_vndf(wo: &Vec3, alpha: f64, u1: f64, u2: f64) -> Vec3 {
    let v = Vec3::new(alpha * wo.x, alpha * wo.y, wo.z).normalize();
    let lensq = v.x * v.x + v.y * v.y;
    let t1 = if lensq > 1e-12 {
        Vec3::new(-v.y, v.x, 0.0) / lensq.sqrt()
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let t2 = v.cross(&t1);
    let r = u1.sqrt();
    let phi = 2.0 * PI * u2;
    let p1 = r * phi.cos();
    let mut p2 = r * phi.sin();
    let s = 0.5 * (1.0 + v.z);
    p2 = (1.0 - s) * (1.0 - p1 * p1).max(0.0).sqrt() + s * p2;
    let nh = t1 * p1 + t2 * p2 + v * (1.0 - p1 * p1 - p2 * p2).max(0.0).sqrt();
    Vec3::new(alpha * nh.x, alpha * nh.y, nh.z.max(1e-9)).normalize()
}

/// pdf of a VNDF-sampled half-vector.
fn ggx_pdf_h(wo: &Vec3, h: &Vec3, alpha: f64) -> f64 {
    let odh = wo.dot(h);
    if odh <= 0.0 || wo.z <= 0.0 {
        return 0.0;
    }
    ggx_g1(wo, alpha) * odh * ggx_d(h, alpha) / wo.z
}

// ---- lobe evaluation ----------------------------------------------------

/// Energy allocated top-down: lobes above the diffuse (specular,
/// clearcoat) take their average reflectance off the layers below,
/// PxrSurface-style, so default materials stay ≤ 1 in the furnace.
fn under_layer_scale(p: &PbrParams) -> f64 {
    let spec_take = 0.2126 * p.specular_f0().x
        + 0.7152 * p.specular_f0().y
        + 0.0722 * p.specular_f0().z;
    let coat_take = 0.04 * p.clearcoat_gain;
    ((1.0 - spec_take) * (1.0 - coat_take)).clamp(0.0, 1.0)
}

fn eval_diffuse(p: &PbrParams, wo: &Vec3, wi: &Vec3) -> Vec3 {
    // Oren-Nayar (sigma in radians ≈ diffuse_roughness).
    let sigma2 = p.diffuse_roughness * p.diffuse_roughness;
    let a = 1.0 - sigma2 / (2.0 * (sigma2 + 0.33));
    let b = 0.45 * sigma2 / (sigma2 + 0.09);
    let sin_o = (1.0 - wo.z * wo.z).max(0.0).sqrt();
    let sin_i = (1.0 - wi.z * wi.z).max(0.0).sqrt();
    let cos_dphi = if sin_o > 1e-6 && sin_i > 1e-6 {
        (((wo.x * wi.x) + (wo.y * wi.y)) / (sin_o * sin_i)).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let (sin_a, tan_b) = if wo.z < wi.z {
        (sin_o, sin_i / wi.z.max(1e-6))
    } else {
        (sin_i, sin_o / wo.z.max(1e-6))
    };
    p.diffuse_color
        * (p.diffuse_gain * under_layer_scale(p) / PI)
        * (a + b * cos_dphi.max(0.0) * sin_a * tan_b)
}

fn eval_fuzz(p: &PbrParams, wo: &Vec3, wi: &Vec3) -> Vec3 {
    // Simple velvet rim: strongest at grazing incidence/exitance.
    let rim = 0.5 * ((1.0 - wi.z).powi(4) + (1.0 - wo.z).powi(4));
    p.fuzz_color * (p.fuzz_gain / PI) * rim
}

fn eval_specular_lobe(f0: Vec3, f90: Vec3, alpha: f64, wo: &Vec3, wi: &Vec3) -> Vec3 {
    let h = (*wo + *wi).normalize();
    let d = ggx_d(&h, alpha);
    let g = ggx_g2(wo, wi, alpha);
    let f = schlick(f0, f90, wo.dot(&h).max(0.0));
    f * (d * g / (4.0 * wo.z * wi.z).max(1e-9))
}

fn spec_pdf(wo: &Vec3, wi: &Vec3, alpha: f64) -> f64 {
    let h = (*wo + *wi).normalize();
    let odh = wo.dot(&h).max(1e-9);
    ggx_pdf_h(wo, &h, alpha) / (4.0 * odh)
}

fn alpha_of(roughness: f64) -> f64 {
    (roughness * roughness).clamp(2.5e-5, 1.0)
}

/// Glass transmission eval + pdf for a refraction direction (wi.z < 0).
/// Returns (f, pdf_given_lobe). PBRT-3 microfacet transmission.
fn glass_transmit(
    p: &PbrParams,
    wo: &Vec3,
    wi: &Vec3,
    eta: f64,
) -> (Vec3, f64) {
    let alpha = alpha_of(p.glass_roughness);
    // Half vector for refraction, oriented to the wo side (z > 0).
    let mut h = (*wo + *wi * eta).normalize();
    if h.z < 0.0 {
        h = -h;
    }
    let odh = wo.dot(&h);
    let idh = wi.dot(&h);
    if odh <= 0.0 || idh >= 0.0 {
        return (Vec3::zero(), 0.0);
    }
    let fres = fresnel_dielectric(odh, eta);
    let d = ggx_d(&h, alpha);
    let g = ggx_g2(wo, &Vec3::new(wi.x, wi.y, -wi.z), alpha);
    let sqrt_denom = odh + eta * idh;
    if sqrt_denom.abs() < 1e-9 {
        return (Vec3::zero(), 0.0);
    }
    // Radiance transport factor 1/eta^2.
    let factor = 1.0 / (eta * eta);
    let f = p.refraction_color
        * (p.glass_gain
            * under_layer_scale(p)
            * (1.0 - fres)
            * d
            * g
            * factor
            * (idh * odh / (wi.z * wo.z)).abs()
            * (eta * eta)
            / (sqrt_denom * sqrt_denom));
    let dwh_dwi = (eta * eta * idh).abs() / (sqrt_denom * sqrt_denom);
    let pdf = ggx_pdf_h(wo, &h, alpha) * dwh_dwi;
    (f, pdf)
}

/// Average Fresnel used for the glass lobe's internal R/T split, computed
/// from the sampled/implied half vector.
fn glass_reflect_prob(wo: &Vec3, h: &Vec3, eta: f64) -> f64 {
    fresnel_dielectric(wo.dot(h).max(0.0), eta)
}

/// Full BSDF eval + pdf. wo.z > 0; wi may be in either hemisphere.
pub fn eval_pdf(p: &PbrParams, wo: &Vec3, wi: &Vec3, eta_rel: f64) -> (Vec3, f64) {
    let Some(w) = p.lobe_weights() else {
        return (Vec3::zero(), 0.0);
    };
    let [wd, ws, wc, wf, wg] = w;

    if wi.z > 0.0 {
        // Reflection side.
        let mut f = Vec3::zero();
        let mut pdf = 0.0;
        let cos_pdf = wi.z / PI;
        if wd > 0.0 {
            f = f + eval_diffuse(p, wo, wi);
            pdf += wd * cos_pdf;
        }
        if wf > 0.0 {
            f = f + eval_fuzz(p, wo, wi);
            pdf += wf * cos_pdf;
        }
        if ws > 0.0 {
            let alpha = alpha_of(p.specular_roughness);
            f = f + eval_specular_lobe(p.specular_f0(), p.specular_edge_color, alpha, wo, wi);
            pdf += ws * spec_pdf(wo, wi, alpha);
        }
        if wc > 0.0 {
            let alpha = alpha_of(p.clearcoat_roughness);
            let f0 = Vec3::new(0.04, 0.04, 0.04) * p.clearcoat_gain;
            f = f + eval_specular_lobe(f0, Vec3::one() * p.clearcoat_gain, alpha, wo, wi);
            pdf += wc * spec_pdf(wo, wi, alpha);
        }
        if wg > 0.0 {
            // Glass reflection branch.
            let alpha = alpha_of(p.glass_roughness);
            let h = (*wo + *wi).normalize();
            let fres = glass_reflect_prob(wo, &h, eta_rel);
            let d = ggx_d(&h, alpha);
            let g = ggx_g2(wo, wi, alpha);
            f = f + Vec3::one()
                * (p.glass_gain * under_layer_scale(p) * fres * d * g
                    / (4.0 * wo.z * wi.z).max(1e-9));
            pdf += wg * fres * spec_pdf(wo, wi, alpha);
        }
        (f, pdf)
    } else {
        // Transmission side: only glass contributes.
        if wg <= 0.0 {
            return (Vec3::zero(), 0.0);
        }
        let (f, pdf_lobe) = glass_transmit(p, wo, wi, eta_rel);
        // pdf = lobe pick * refract-branch probability * (pdf_h * jacobian),
        // matching the procedure in sample().
        let mut h = (*wo + *wi * eta_rel).normalize();
        if h.z < 0.0 {
            h = -h;
        }
        let fres = glass_reflect_prob(wo, &h, eta_rel);
        (f, wg * (1.0 - fres) * pdf_lobe)
    }
}

fn cosine_sample_local(rng: &mut Pcg32) -> Vec3 {
    let (u, v) = rng.next_2d();
    let r = u.sqrt();
    let phi = 2.0 * PI * v;
    Vec3::new(r * phi.cos(), r * phi.sin(), (1.0 - u).max(0.0).sqrt())
}

/// Sample the composite BSDF. wo.z > 0.
pub fn sample(p: &PbrParams, wo: &Vec3, eta_rel: f64, rng: &mut Pcg32) -> Option<BsdfSample> {
    let w = p.lobe_weights()?;
    let [wd, ws, wc, wf, wg] = w;

    let pick = rng.next_f64();
    let mut transmitted = false;
    let wi = if pick < wd + wf {
        // Diffuse / fuzz share cosine sampling.
        cosine_sample_local(rng)
    } else if pick < wd + wf + ws {
        let alpha = alpha_of(p.specular_roughness);
        let (u1, u2) = rng.next_2d();
        let h = ggx_sample_vndf(wo, alpha, u1, u2);
        let wi = (-*wo).reflect(&h);
        if wi.z <= 0.0 {
            return None;
        }
        wi
    } else if pick < wd + wf + ws + wc {
        let alpha = alpha_of(p.clearcoat_roughness);
        let (u1, u2) = rng.next_2d();
        let h = ggx_sample_vndf(wo, alpha, u1, u2);
        let wi = (-*wo).reflect(&h);
        if wi.z <= 0.0 {
            return None;
        }
        wi
    } else if wg > 0.0 {
        // Glass: pick reflect vs refract by exact Fresnel at the sampled h.
        let alpha = alpha_of(p.glass_roughness);
        let (u1, u2) = rng.next_2d();
        let h = ggx_sample_vndf(wo, alpha, u1, u2);
        let fres = glass_reflect_prob(wo, &h, eta_rel);
        if rng.next_f64() < fres {
            let wi = (-*wo).reflect(&h);
            if wi.z <= 0.0 {
                return None;
            }
            wi
        } else {
            // Refract wo through h.
            let cos_i = wo.dot(&h);
            let sin2_t = (1.0 - cos_i * cos_i) / (eta_rel * eta_rel);
            if sin2_t >= 1.0 {
                return None; // TIR handled by fres≈1 anyway
            }
            let cos_t = (1.0 - sin2_t).sqrt();
            let wi = (-*wo) / eta_rel + h * (cos_i / eta_rel - cos_t);
            let wi = wi.normalize();
            if wi.z >= 0.0 {
                return None;
            }
            transmitted = true;
            wi
        }
    } else {
        return None;
    };

    let (f, pdf) = eval_pdf(p, wo, &wi, eta_rel);
    if pdf <= 0.0 {
        return None;
    }
    Some(BsdfSample { wi, f, pdf, transmitted })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params_diffuse() -> PbrParams {
        PbrParams {
            diffuse_gain: 1.0,
            diffuse_color: Vec3::one(),
            specular_ior: 1.0,
            ..Default::default()
        }
    }

    /// The composite pdf must integrate to ~1 over the sphere.
    #[test]
    fn pdf_integrates_to_one() {
        let cases = [
            params_diffuse(),
            PbrParams::from_metal(Vec3::new(0.9, 0.7, 0.4), 0.3),
            PbrParams::from_plastic(Vec3::new(0.5, 0.2, 0.2), 0.15),
        ];
        let wo = Vec3::new(0.3, -0.15, 0.94).normalize();
        for (ci, p) in cases.iter().enumerate() {
            let mut sum = 0.0;
            let n_theta = 64;
            let n_phi = 128;
            for it in 0..n_theta {
                // reflection hemisphere only (no glass in these cases)
                let theta = (it as f64 + 0.5) / n_theta as f64 * (PI / 2.0);
                for ip in 0..n_phi {
                    let phi = (ip as f64 + 0.5) / n_phi as f64 * 2.0 * PI;
                    let wi = Vec3::new(
                        theta.sin() * phi.cos(),
                        theta.sin() * phi.sin(),
                        theta.cos(),
                    );
                    let (_, pdf) = eval_pdf(p, &wo, &wi, 1.5);
                    sum += pdf * theta.sin() * (PI / 2.0 / n_theta as f64)
                        * (2.0 * PI / n_phi as f64);
                }
            }
            assert!(
                (sum - 1.0).abs() < 0.05,
                "case {ci}: pdf integrates to {sum:.4}, expected ~1"
            );
        }
    }

    /// Monte Carlo white-furnace check per lobe: E[f cos / pdf] under BSDF
    /// sampling must be <= 1 (energy conservation) and near 1 for an
    /// albedo-1 diffuse.
    #[test]
    fn furnace_estimates() {
        let mut rng = Pcg32::new(7, 11);
        let wo = Vec3::new(0.2, 0.1, 0.97).normalize();
        let run = |p: &PbrParams, rng: &mut Pcg32| -> f64 {
            let n = 200_000;
            let mut acc = 0.0;
            for _ in 0..n {
                if let Some(s) = sample(p, &wo, 1.5, rng) {
                    acc += lum(&s.f) * s.wi.z.abs() / s.pdf;
                }
            }
            acc / n as f64
        };
        let diffuse = run(&params_diffuse(), &mut rng);
        assert!((diffuse - 1.0).abs() < 0.02, "diffuse albedo {diffuse:.4}");

        let metal = run(&PbrParams::from_metal(Vec3::one(), 0.4), &mut rng);
        assert!(metal <= 1.02, "metal energy {metal:.4} > 1");
        assert!(metal > 0.75, "rough metal lost too much energy: {metal:.4}");

        let plastic = run(&PbrParams::from_plastic(Vec3::one(), 0.2), &mut rng);
        assert!(plastic <= 1.03, "plastic energy {plastic:.4} > 1");
    }

    /// BSDF-sampled and uniform-sampled estimates of the reflected energy
    /// must agree (validates sample()/pdf() consistency).
    #[test]
    fn estimator_consistency() {
        let p = PbrParams::from_metal(Vec3::new(0.8, 0.6, 0.3), 0.35);
        let wo = Vec3::new(-0.3, 0.2, 0.93).normalize();
        let mut rng = Pcg32::new(3, 9);

        let n = 400_000;
        let mut by_bsdf = 0.0;
        for _ in 0..n {
            if let Some(s) = sample(&p, &wo, 1.5, &mut rng) {
                by_bsdf += lum(&s.f) * s.wi.z.abs() / s.pdf;
            }
        }
        by_bsdf /= n as f64;

        let mut by_uniform = 0.0;
        for _ in 0..n {
            // uniform hemisphere
            let (u, v) = rng.next_2d();
            let z = u;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let phi = 2.0 * PI * v;
            let wi = Vec3::new(r * phi.cos(), r * phi.sin(), z);
            let (f, _) = eval_pdf(&p, &wo, &wi, 1.5);
            by_uniform += lum(&f) * wi.z * 2.0 * PI;
        }
        by_uniform /= n as f64;

        let rel = (by_bsdf - by_uniform).abs() / by_uniform.max(1e-6);
        assert!(
            rel < 0.03,
            "estimators disagree: bsdf {by_bsdf:.4} vs uniform {by_uniform:.4} ({rel:.4})"
        );
    }
}
