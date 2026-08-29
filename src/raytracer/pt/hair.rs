//! Energy-conserving hair scattering (roadmap Phase 8), after
//! Marschner et al. 2003 as reformulated by d'Eon et al. 2011 and the
//! pbrt-v3 implementation: R / TT / TRT lobes plus a residual, with
//! energy-conserving longitudinal gaussians (Mp), trimmed-logistic
//! azimuthal profiles (Np), and Fresnel/absorption attenuation (Ap).
//!
//! Convention difference from pbrt: our `f` does NOT carry pbrt's
//! 1/|cos(theta_i)| factor, and the integrator does not multiply hair
//! contributions by a surface cosine — f is the per-solid-angle
//! scattering directly, so `∫ f dω = 1` at zero absorption (the furnace
//! test asserts exactly this).
//!
//! Directions are expressed in the fiber frame: x = tangent (fiber axis),
//! yz = the normal plane; theta is measured from the normal plane, phi
//! around the fiber.

use super::sampler::Pcg32;
use crate::math::Vec3;
use crate::parser::ParamList;
use std::f64::consts::PI;

const P_MAX: usize = 3;
const SQRT_PI_OVER_8: f64 = 0.626657069;

#[derive(Debug, Clone)]
pub struct HairParams {
    /// Absorption per unit fiber-diameter path length.
    pub sigma_a: Vec3,
    /// Longitudinal roughness in [0,1].
    pub beta_m: f64,
    /// Azimuthal roughness in [0,1].
    pub beta_n: f64,
    pub eta: f64,
    /// Precomputed longitudinal variances per lobe.
    v: [f64; P_MAX + 1],
    /// Precomputed azimuthal logistic scale.
    s: f64,
}

impl HairParams {
    pub fn new(sigma_a: Vec3, beta_m: f64, beta_n: f64, eta: f64) -> Self {
        let beta_m = beta_m.clamp(0.01, 1.0);
        let beta_n = beta_n.clamp(0.01, 1.0);
        let v0 = (0.726 * beta_m + 0.812 * beta_m * beta_m + 3.7 * beta_m.powi(20)).powi(2);
        let v = [v0, 0.25 * v0, 4.0 * v0, 4.0 * v0];
        let s = SQRT_PI_OVER_8
            * (0.265 * beta_n + 1.194 * beta_n * beta_n + 5.372 * beta_n.powi(22));
        Self { sigma_a, beta_m, beta_n, eta, v, s }
    }

    /// Longitudinal variances per lobe (for GPU export).
    pub fn lobe_variances(&self) -> [f64; P_MAX + 1] {
        self.v
    }

    /// Azimuthal logistic scale (for GPU export).
    pub fn azimuthal_s(&self) -> f64 {
        self.s
    }

    /// Absorption coefficient that yields the given fiber color under
    /// azimuthal roughness beta_n (pbrt's SigmaAFromReflectance).
    pub fn sigma_a_from_color(c: Vec3, beta_n: f64) -> Vec3 {
        let denom = 5.969 - 0.215 * beta_n + 2.532 * beta_n.powi(2)
            - 10.73 * beta_n.powi(3)
            + 5.574 * beta_n.powi(4)
            + 0.245 * beta_n.powi(5);
        let f = |x: f64| {
            let x = x.clamp(1e-4, 0.999);
            (x.ln() / denom).powi(2)
        };
        Vec3::new(f(c.x), f(c.y), f(c.z))
    }

    /// Parse `Bxdf "PxrMarschnerHair"` parameters.
    pub fn from_bxdf_params(params: &ParamList<'_>) -> Self {
        let beta_m = params.get_number("roughness").unwrap_or(0.3);
        let beta_n = params.get_number("azimuthalRoughness").unwrap_or(0.3);
        let eta = params.get_number("ior").unwrap_or(1.55);
        let sigma_a = if let Some(a) = params.get_numbers("absorption") {
            if a.len() >= 3 {
                Vec3::new(a[0], a[1], a[2])
            } else {
                Vec3::new(a[0], a[0], a[0])
            }
        } else {
            let c = params
                .get_numbers("color")
                .and_then(|v| (v.len() >= 3).then(|| Vec3::new(v[0], v[1], v[2])))
                .unwrap_or(Vec3::new(0.35, 0.2, 0.06));
            Self::sigma_a_from_color(c, beta_n.clamp(0.01, 1.0))
        };
        Self::new(sigma_a, beta_m, beta_n, eta)
    }
}

// ---- numerics ----------------------------------------------------------

fn i0(x: f64) -> f64 {
    let mut val = 0.0;
    let mut x2i = 1.0;
    let mut ifact = 1.0;
    let mut i4 = 1.0;
    for i in 0..10 {
        if i > 1 {
            ifact *= i as f64;
        }
        val += x2i / (i4 * ifact * ifact);
        x2i *= x * x;
        i4 *= 4.0;
    }
    val
}

fn log_i0(x: f64) -> f64 {
    if x > 12.0 {
        x + 0.5 * (-(2.0 * PI * x).ln() + 1.0 / (8.0 * x))
    } else {
        i0(x).ln()
    }
}

/// Energy-conserving longitudinal lobe (d'Eon 2011).
fn mp(cos_ti: f64, cos_to: f64, sin_ti: f64, sin_to: f64, v: f64) -> f64 {
    let a = cos_ti * cos_to / v;
    let b = sin_ti * sin_to / v;
    if v <= 0.1 {
        (log_i0(a) - b - 1.0 / v + 0.6931 + (1.0 / (2.0 * v)).ln()).exp()
    } else {
        ((-b).exp() * i0(a)) / ((1.0 / v).sinh() * 2.0 * v)
    }
}

fn fr_dielectric(mut cos_i: f64, eta_i: f64, eta_t: f64) -> f64 {
    cos_i = cos_i.clamp(-1.0, 1.0);
    let (eta_i, eta_t, cos_i) = if cos_i > 0.0 {
        (eta_i, eta_t, cos_i)
    } else {
        (eta_t, eta_i, -cos_i)
    };
    let sin_i = (1.0 - cos_i * cos_i).max(0.0).sqrt();
    let sin_t = eta_i / eta_t * sin_i;
    if sin_t >= 1.0 {
        return 1.0;
    }
    let cos_t = (1.0 - sin_t * sin_t).max(0.0).sqrt();
    let rparl = (eta_t * cos_i - eta_i * cos_t) / (eta_t * cos_i + eta_i * cos_t);
    let rperp = (eta_i * cos_i - eta_t * cos_t) / (eta_i * cos_i + eta_t * cos_t);
    (rparl * rparl + rperp * rperp) / 2.0
}

/// Attenuation per lobe: Fresnel at entry, transmission body, exits.
fn ap(cos_to: f64, eta: f64, h: f64, t: Vec3) -> [Vec3; P_MAX + 1] {
    let cos_gamma_o = (1.0 - h * h).max(0.0).sqrt();
    let cos_theta = cos_to * cos_gamma_o;
    let f = fr_dielectric(cos_theta, 1.0, eta);
    let mut out = [Vec3::zero(); P_MAX + 1];
    out[0] = Vec3::new(f, f, f);
    out[1] = t * ((1.0 - f) * (1.0 - f));
    for p in 2..P_MAX {
        out[p] = Vec3::new(
            out[p - 1].x * t.x * f,
            out[p - 1].y * t.y * f,
            out[p - 1].z * t.z * f,
        );
    }
    // Residual: geometric series of the remaining bounces.
    let last = out[P_MAX - 1];
    out[P_MAX] = Vec3::new(
        last.x * f * t.x / (1.0 - t.x * f).max(1e-6),
        last.y * f * t.y / (1.0 - t.y * f).max(1e-6),
        last.z * f * t.z / (1.0 - t.z * f).max(1e-6),
    );
    out
}

fn logistic(x: f64, s: f64) -> f64 {
    let x = x.abs();
    let e = (-x / s).exp();
    e / (s * (1.0 + e) * (1.0 + e))
}

fn logistic_cdf(x: f64, s: f64) -> f64 {
    1.0 / (1.0 + (-x / s).exp())
}

fn trimmed_logistic(x: f64, s: f64, a: f64, b: f64) -> f64 {
    logistic(x, s) / (logistic_cdf(b, s) - logistic_cdf(a, s))
}

fn sample_trimmed_logistic(u: f64, s: f64, a: f64, b: f64) -> f64 {
    let k = logistic_cdf(b, s) - logistic_cdf(a, s);
    let x = -s * (1.0 / (u * k + logistic_cdf(a, s)).clamp(1e-9, 1.0 - 1e-9) - 1.0).ln();
    x.clamp(a, b)
}

/// Net azimuthal deflection for lobe p.
fn phi_fn(p: usize, gamma_o: f64, gamma_t: f64) -> f64 {
    2.0 * p as f64 * gamma_t - 2.0 * gamma_o + p as f64 * PI
}

fn np(phi: f64, p: usize, s: f64, gamma_o: f64, gamma_t: f64) -> f64 {
    let mut dphi = phi - phi_fn(p, gamma_o, gamma_t);
    while dphi > PI {
        dphi -= 2.0 * PI;
    }
    while dphi < -PI {
        dphi += 2.0 * PI;
    }
    trimmed_logistic(dphi, s, -PI, PI)
}

// ---- shared per-evaluation geometry ------------------------------------

struct HairGeom {
    sin_to: f64,
    cos_to: f64,
    phi_o: f64,
    gamma_o: f64,
    gamma_t: f64,
    /// Body transmittance for this incidence.
    t: Vec3,
}

fn geom(params: &HairParams, wo: &Vec3, h: f64) -> HairGeom {
    let sin_to = wo.x.clamp(-1.0, 1.0);
    let cos_to = (1.0 - sin_to * sin_to).max(0.0).sqrt();
    let phi_o = wo.z.atan2(wo.y);
    let gamma_o = h.clamp(-1.0, 1.0).asin();
    // Refracted ray inside the fiber.
    let sin_tt = sin_to / params.eta;
    let cos_tt = (1.0 - sin_tt * sin_tt).max(0.0).sqrt();
    let etap = (params.eta * params.eta - sin_to * sin_to).max(0.0).sqrt() / cos_to.max(1e-9);
    let sin_gamma_t = (h / etap).clamp(-1.0, 1.0);
    let cos_gamma_t = (1.0 - sin_gamma_t * sin_gamma_t).max(0.0).sqrt();
    let gamma_t = sin_gamma_t.asin();
    let l = 2.0 * cos_gamma_t / cos_tt.max(1e-9);
    let t = Vec3::new(
        (-params.sigma_a.x * l).exp(),
        (-params.sigma_a.y * l).exp(),
        (-params.sigma_a.z * l).exp(),
    );
    HairGeom { sin_to, cos_to, phi_o, gamma_o, gamma_t, t }
}

/// BSDF value for (wo, wi) in the fiber frame with azimuthal offset h.
/// Per-solid-angle; no cosine factors on either side.
pub fn f(params: &HairParams, wo: &Vec3, wi: &Vec3, h: f64) -> Vec3 {
    let g = geom(params, wo, h);
    let sin_ti = wi.x.clamp(-1.0, 1.0);
    let cos_ti = (1.0 - sin_ti * sin_ti).max(0.0).sqrt();
    let phi_i = wi.z.atan2(wi.y);
    let phi = phi_i - g.phi_o;
    let a = ap(g.cos_to, params.eta, h, g.t);
    let mut out = Vec3::zero();
    for (p, apv) in a.iter().enumerate() {
        let m = mp(cos_ti, g.cos_to, sin_ti, g.sin_to, params.v[p]);
        let n = if p < P_MAX {
            np(phi, p, params.s, g.gamma_o, g.gamma_t)
        } else {
            1.0 / (2.0 * PI)
        };
        out = out + *apv * (m * n);
    }
    out
}

/// Per-lobe selection weights (luminance of Ap, normalized).
fn ap_pdf(params: &HairParams, g: &HairGeom, h: f64) -> [f64; P_MAX + 1] {
    let a = ap(g.cos_to, params.eta, h, g.t);
    let lum = |c: &Vec3| 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
    let mut w = [0.0; P_MAX + 1];
    let mut total = 0.0;
    for (p, apv) in a.iter().enumerate() {
        w[p] = lum(apv);
        total += w[p];
    }
    if total > 1e-12 {
        for wp in &mut w {
            *wp /= total;
        }
    } else {
        w[0] = 1.0;
    }
    w
}

/// Solid-angle pdf of `wi` given `wo` (matches sample()).
pub fn pdf(params: &HairParams, wo: &Vec3, wi: &Vec3, h: f64) -> f64 {
    let g = geom(params, wo, h);
    let sin_ti = wi.x.clamp(-1.0, 1.0);
    let cos_ti = (1.0 - sin_ti * sin_ti).max(0.0).sqrt();
    let phi_i = wi.z.atan2(wi.y);
    let phi = phi_i - g.phi_o;
    let w = ap_pdf(params, &g, h);
    let mut out = 0.0;
    for (p, wp) in w.iter().enumerate() {
        let m = mp(cos_ti, g.cos_to, sin_ti, g.sin_to, params.v[p]);
        let n = if p < P_MAX {
            np(phi, p, params.s, g.gamma_o, g.gamma_t)
        } else {
            1.0 / (2.0 * PI)
        };
        out += wp * m * n;
    }
    out.max(0.0)
}

/// Importance-sample an outgoing direction. Returns (wi, f, pdf).
pub fn sample(
    params: &HairParams,
    wo: &Vec3,
    h: f64,
    rng: &mut Pcg32,
) -> Option<(Vec3, Vec3, f64)> {
    let g = geom(params, wo, h);
    let w = ap_pdf(params, &g, h);

    // Pick a lobe by attenuation energy.
    let mut u = rng.next_f64();
    let mut p = P_MAX;
    for (i, wp) in w.iter().enumerate() {
        if u < *wp {
            p = i;
            break;
        }
        u -= wp;
    }

    // Longitudinal: sample Mp's cone (d'Eon's exact inverse).
    let v = params.v[p.min(P_MAX)];
    let u1 = rng.next_f64().max(1e-9);
    let u2 = rng.next_f64();
    let cos_theta = 1.0 + v * (u1 + (1.0 - u1) * (-2.0 / v).exp()).ln();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let cos_phi_l = (2.0 * PI * u2).cos();
    // Rotate about wo's longitudinal angle.
    let sin_ti = -cos_theta * g.sin_to + sin_theta * cos_phi_l * g.cos_to;
    let cos_ti = (1.0 - sin_ti * sin_ti).max(0.0).sqrt();

    // Azimuthal.
    let u3 = rng.next_f64();
    let dphi = if p < P_MAX {
        phi_fn(p, g.gamma_o, g.gamma_t) + sample_trimmed_logistic(u3, params.s, -PI, PI)
    } else {
        2.0 * PI * u3
    };
    let phi_i = g.phi_o + dphi;

    let wi = Vec3::new(sin_ti, cos_ti * phi_i.cos(), cos_ti * phi_i.sin());
    let pdf_v = pdf(params, wo, &wi, h);
    if pdf_v <= 1e-12 {
        return None;
    }
    let f_v = f(params, wo, &wi, h);
    Some((wi, f_v, pdf_v))
}

/// The fiber shading frame at a curve hit: x = tangent; the impact
/// parameter h in [-1,1] falls out of the geometric normal.
pub struct FiberFrame {
    pub x: Vec3,
    pub y: Vec3,
    pub z: Vec3,
    pub h: f64,
}

impl FiberFrame {
    /// `tangent` along the fiber, `n` the capsule's geometric normal at
    /// the hit, `wo` the world-space direction back toward the viewer.
    pub fn new(tangent: &Vec3, n: &Vec3, wo: &Vec3) -> Self {
        let x = tangent.normalize();
        // Reference azimuth: the viewer direction projected off the fiber.
        let mut z = *wo - x * wo.dot(&x);
        if z.length_squared() < 1e-12 {
            z = *n - x * n.dot(&x);
        }
        let z = z.normalize();
        let y = z.cross(&x).normalize();
        // Impact parameter from the normal: h = sin(gamma_o), the offset
        // of the grazing ray across the fiber's width.
        let n_perp = (*n - x * n.dot(&x)).normalize();
        let h = n_perp.dot(&y).clamp(-1.0, 1.0);
        Self { x, y, z, h }
    }

    pub fn to_local(&self, w: &Vec3) -> Vec3 {
        Vec3::new(w.dot(&self.x), w.dot(&self.y), w.dot(&self.z))
    }

    pub fn to_world(&self, w: &Vec3) -> Vec3 {
        self.x * w.x + self.y * w.y + self.z * w.z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With zero absorption the fiber redistributes all energy:
    /// ∫ f dω ≈ 1 for any wo.
    #[test]
    fn white_furnace() {
        let params = HairParams::new(Vec3::zero(), 0.3, 0.3, 1.55);
        let mut rng = Pcg32::for_pixel_sample(7, 3);
        for (sin_to, h) in [(0.0, 0.0), (0.4, 0.3), (-0.7, -0.6), (0.2, 0.9)] {
            let cos_to = (1.0f64 - sin_to * sin_to).sqrt();
            let wo = Vec3::new(sin_to, cos_to, 0.0);
            let n = 40_000;
            let mut sum = Vec3::zero();
            for _ in 0..n {
                // Uniform sphere sample.
                let u1 = rng.next_f64();
                let u2 = rng.next_f64();
                let z = 1.0 - 2.0 * u1;
                let r = (1.0f64 - z * z).max(0.0).sqrt();
                let phi = 2.0 * PI * u2;
                let wi = Vec3::new(z, r * phi.cos(), r * phi.sin());
                sum = sum + f(&params, &wo, &wi, h) * (4.0 * PI);
            }
            let mean = sum / n as f64;
            assert!(
                (mean.x - 1.0).abs() < 0.05,
                "furnace failed at sin_to={sin_to} h={h}: {mean:?}"
            );
        }
    }

    /// Sampling must agree with pdf(): E[f/pdf] over samples equals the
    /// furnace integral (~1 at zero absorption).
    #[test]
    fn sampling_consistency() {
        let params = HairParams::new(Vec3::zero(), 0.35, 0.4, 1.55);
        let mut rng = Pcg32::for_pixel_sample(11, 5);
        let wo = Vec3::new(0.3, (1.0f64 - 0.09).sqrt(), 0.0);
        let h = -0.4;
        let n = 40_000;
        let mut sum = 0.0;
        for _ in 0..n {
            if let Some((_, f_v, pdf_v)) = sample(&params, &wo, h, &mut rng) {
                sum += (0.2126 * f_v.x + 0.7152 * f_v.y + 0.0722 * f_v.z) / pdf_v;
            }
        }
        let mean = sum / n as f64;
        assert!((mean - 1.0).abs() < 0.05, "E[f/pdf] = {mean}");
    }

    /// Absorption darkens TT/TRT (forward/secondary lobes) but leaves the
    /// surface reflection R intact.
    #[test]
    fn absorption_darkens() {
        let white = HairParams::new(Vec3::zero(), 0.3, 0.3, 1.55);
        let dark = HairParams::new(Vec3::new(3.0, 3.0, 3.0), 0.3, 0.3, 1.55);
        let wo = Vec3::new(0.0, 1.0, 0.0);
        // Transmission direction (through the fiber).
        let wi_tt = Vec3::new(0.0, -1.0, 0.0);
        let fw = f(&white, &wo, &wi_tt, 0.0);
        let fd = f(&dark, &wo, &wi_tt, 0.0);
        assert!(fd.x < fw.x * 0.2, "absorption failed: {} vs {}", fd.x, fw.x);
    }

    #[test]
    fn fiber_frame_h_spans_width() {
        let t = Vec3::new(1.0, 0.0, 0.0);
        let wo = Vec3::new(0.0, 0.0, 1.0); // looking down +z at the fiber
        // Normal straight back at the viewer: center hit, h ~ 0.
        let center = FiberFrame::new(&t, &Vec3::new(0.0, 0.0, 1.0), &wo);
        assert!(center.h.abs() < 1e-9, "h = {}", center.h);
        // Normal to the side: edge graze, |h| ~ 1.
        let edge = FiberFrame::new(&t, &Vec3::new(0.0, 1.0, 0.0), &wo);
        assert!(edge.h.abs() > 0.99, "h = {}", edge.h);
    }
}
