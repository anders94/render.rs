//! Volume transport helpers for the path tracer (roadmap Phase 10):
//! spectral single-sample distance sampling in homogeneous media,
//! weighted delta tracking in heterogeneous media, and ratio-tracked
//! transmittance. All estimators are unbiased; RGB extinction uses the
//! one-sample spectral MIS weighting (value / average-channel pdf).

use super::sampler::Pcg32;
use crate::math::{Point3, Vec3};
use crate::raytracer::Ray;
use crate::scene::Medium;

/// Result of propagating a ray segment through a medium.
pub enum MediumEvent {
    /// Scattered at distance t; `weight` folds transmittance, sigma_s and
    /// the sampling pdf.
    Scatter { t: f64, weight: Vec3 },
    /// Reached the segment end; `weight` folds transmittance and the
    /// pass-through probability.
    Pass { weight: Vec3 },
}

fn avg(v: &Vec3) -> f64 {
    (v.x + v.y + v.z) / 3.0
}

fn exp3(v: &Vec3, t: f64) -> Vec3 {
    Vec3::new((-v.x * t).exp(), (-v.y * t).exp(), (-v.z * t).exp())
}

/// Sample a scattering distance along [0, t_max] of `ray` inside `medium`.
/// `beta` is the path throughput — the sampling channel is chosen
/// proportional to it (spectral one-sample MIS), which tames the chroma
/// noise of media whose extinction differs strongly per channel.
pub fn sample_distance(
    medium: &Medium,
    ray: &Ray,
    t_max: f64,
    beta: &Vec3,
    rng: &mut Pcg32,
) -> MediumEvent {
    // Media are slabs of finite extent: beyond max_distance the segment
    // is free — sample within the clamped range; the tail passes with the
    // clamped transmittance.
    let t_max = t_max.min(medium.max_distance);
    if medium.is_homogeneous() {
        return sample_homogeneous(medium, t_max, beta, rng);
    }
    sample_delta_tracked(medium, ray, t_max, rng)
}

/// Channel weights proportional to throughput (uniform fallback).
fn channel_weights(beta: &Vec3) -> [f64; 3] {
    let b = Vec3::new(beta.x.max(0.0), beta.y.max(0.0), beta.z.max(0.0));
    let total = b.x + b.y + b.z;
    if total <= 1e-12 {
        return [1.0 / 3.0; 3];
    }
    [b.x / total, b.y / total, b.z / total]
}

fn sample_homogeneous(
    medium: &Medium,
    t_max: f64,
    beta: &Vec3,
    rng: &mut Pcg32,
) -> MediumEvent {
    let st = medium.sigma_t();
    let st_avg = avg(&st);
    if st_avg <= 1e-9 {
        return MediumEvent::Pass { weight: Vec3::one() };
    }
    let w = channel_weights(beta);
    let channels = [st.x, st.y, st.z];
    // Pick a channel by throughput weight.
    let u = rng.next_f64();
    let c = if u < w[0] {
        0
    } else if u < w[0] + w[1] {
        1
    } else {
        2
    };
    let sigma_c = channels[c].max(1e-9);
    let t = -(1.0 - rng.next_f64()).max(1e-12).ln() / sigma_c;
    if t < t_max {
        // pdf(t) = sum_c w_c sigma_c e^{-sigma_c t}
        let tr = exp3(&st, t);
        let pdf = w[0] * st.x * tr.x + w[1] * st.y * tr.y + w[2] * st.z * tr.z;
        if pdf <= 1e-300 {
            return MediumEvent::Pass { weight: Vec3::zero() };
        }
        MediumEvent::Scatter { t, weight: medium.sigma_s * tr / pdf }
    } else {
        // pdf(pass) = sum_c w_c e^{-sigma_c t_max}
        let tr = exp3(&st, t_max);
        let pdf = w[0] * tr.x + w[1] * tr.y + w[2] * tr.z;
        if pdf <= 1e-300 {
            return MediumEvent::Pass { weight: Vec3::zero() };
        }
        MediumEvent::Pass { weight: tr / pdf }
    }
}

/// Weighted delta tracking against the channel-max majorant. Density is a
/// scalar field, so per-channel extinction is sigma_t * rho(p).
fn sample_delta_tracked(
    medium: &Medium,
    ray: &Ray,
    t_max: f64,
    rng: &mut Pcg32,
) -> MediumEvent {
    let majorant = medium.majorant();
    if majorant <= 1e-9 {
        return MediumEvent::Pass { weight: Vec3::one() };
    }
    let st = medium.sigma_t();
    let mut t = 0.0;
    let mut weight = Vec3::one();
    for _ in 0..10_000 {
        t -= (1.0 - rng.next_f64()).max(1e-12).ln() / majorant;
        if t >= t_max {
            return MediumEvent::Pass { weight };
        }
        let p = ray.at(t);
        let rho = medium.density_at(&p);
        let st_here = st * rho;
        // Probability of treating this tentative collision as real.
        let p_real = (avg(&st_here) / majorant).clamp(0.0, 1.0);
        if rng.next_f64() < p_real {
            let sigma_s_here = medium.sigma_s * rho;
            return MediumEvent::Scatter {
                t,
                weight: weight * sigma_s_here / (majorant * p_real),
            };
        }
        // Null collision: component-wise correction.
        let p_null = (1.0 - p_real).max(1e-9);
        weight = weight
            * Vec3::new(
                (1.0 - st_here.x / majorant) / p_null,
                (1.0 - st_here.y / majorant) / p_null,
                (1.0 - st_here.z / majorant) / p_null,
            );
        if weight.x.max(weight.y).max(weight.z) < 1e-6 {
            return MediumEvent::Pass { weight: Vec3::zero() };
        }
    }
    MediumEvent::Pass { weight }
}

/// Transmittance of `medium` along a segment (ratio tracking for
/// heterogeneous, closed form for homogeneous).
pub fn transmittance(
    medium: &Medium,
    origin: &Point3,
    dir: &Vec3,
    dist: f64,
    rng: &mut Pcg32,
) -> Vec3 {
    let dist = dist.min(medium.max_distance);
    let st = medium.sigma_t();
    if medium.is_homogeneous() {
        return exp3(&st, dist);
    }
    let majorant = medium.majorant();
    if majorant <= 1e-9 {
        return Vec3::one();
    }
    let mut t = 0.0;
    let mut tr = Vec3::one();
    for _ in 0..10_000 {
        t -= (1.0 - rng.next_f64()).max(1e-12).ln() / majorant;
        if t >= dist {
            break;
        }
        let p = *origin + *dir * t;
        let rho = medium.density_at(&p);
        tr = tr
            * Vec3::new(
                1.0 - st.x * rho / majorant,
                1.0 - st.y * rho / majorant,
                1.0 - st.z * rho / majorant,
            );
        if tr.x.max(tr.y).max(tr.z) < 1e-5 {
            return Vec3::zero();
        }
    }
    tr
}

/// Equiangular distance sampling (Kulla-Fajardo 2012): place a sample
/// along [0, t_max] of the ray with density proportional to 1/r² toward
/// a point light — exactly where single-scatter glow concentrates.
/// Returns (t, pdf).
pub fn equiangular_sample(
    origin: &Point3,
    dir: &Vec3,
    t_max: f64,
    light_pos: &Point3,
    u: f64,
) -> (f64, f64) {
    let delta = (*light_pos - *origin).dot(dir);
    let h = (*light_pos - (*origin + *dir * delta)).length().max(1e-6);
    let theta_a = (-delta / h).atan();
    let theta_b = ((t_max - delta) / h).atan();
    let span = (theta_b - theta_a).max(1e-9);
    let t = delta + h * (theta_a + u * span).tan();
    let t = t.clamp(0.0, t_max);
    let pdf = h / (span * (h * h + (t - delta) * (t - delta)));
    (t, pdf)
}

/// pdf of equiangular_sample at distance t (for MIS weights).
pub fn equiangular_pdf(
    origin: &Point3,
    dir: &Vec3,
    t_max: f64,
    light_pos: &Point3,
    t: f64,
) -> f64 {
    let delta = (*light_pos - *origin).dot(dir);
    let h = (*light_pos - (*origin + *dir * delta)).length().max(1e-6);
    let theta_a = (-delta / h).atan();
    let theta_b = ((t_max - delta) / h).atan();
    let span = (theta_b - theta_a).max(1e-9);
    h / (span * (h * h + (t - delta) * (t - delta)))
}

/// Approximate analytic pdf of the medium's distance sampling at t (used
/// ONLY for MIS weighting against equiangular — approximate weights are
/// still unbiased as long as the pair sums to one, which these do).
pub fn distance_pdf_approx(medium: &Medium, t: f64, t_max: f64) -> f64 {
    let st = medium.sigma_t();
    let s = ((st.x + st.y + st.z) / 3.0).max(1e-9);
    if t < t_max {
        s * (-s * t).exp()
    } else {
        (-s * t_max).exp()
    }
}

/// Random-walk medium for subsurface scattering, from PxrSurface's
/// artist parameters. Two fits do the heavy lifting:
/// - albedo inversion (Kulla-Conty): the walk's single-scatter albedo
///   that makes a semi-infinite isotropic medium *look like* surface
///   color A — much higher than A, since multiple scattering darkens
///   (alpha = 1 - e^{-A(11.43 - 15.38A + 13.91A²)});
/// - Christensen-Burley scaling: sigma_t = s(A)/dmfp with
///   s = 1.85 - A + 7|A - 0.8|³, tying the mean free path to dmfp.
pub fn sss_medium(color: &Vec3, dmfp: &Vec3) -> Medium {
    let alpha_fit = |a: f64| {
        let a = a.clamp(0.0, 1.0);
        1.0 - (-a * (11.43 - 15.38 * a + 13.91 * a * a)).exp()
    };
    let s_fit = |a: f64| 1.85 - a + 7.0 * (a - 0.8).abs().powi(3);
    let a = Vec3::new(
        color.x.clamp(0.0, 1.0),
        color.y.clamp(0.0, 1.0),
        color.z.clamp(0.0, 1.0),
    );
    let st = Vec3::new(
        s_fit(a.x) / dmfp.x.max(1e-5),
        s_fit(a.y) / dmfp.y.max(1e-5),
        s_fit(a.z) / dmfp.z.max(1e-5),
    );
    let alpha = Vec3::new(alpha_fit(a.x), alpha_fit(a.y), alpha_fit(a.z));
    let ss = alpha * st;
    Medium {
        sigma_a: st - ss,
        sigma_s: ss,
        g: 0.0,
        density: None,
        emission: crate::math::Vec3::zero(),
        max_distance: f64::INFINITY,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::DensityField;

    fn fog(sigma_s: f64, sigma_a: f64) -> Medium {
        Medium {
            sigma_a: Vec3::new(sigma_a, sigma_a, sigma_a),
            sigma_s: Vec3::new(sigma_s, sigma_s, sigma_s),
            g: 0.0,
            density: None,
            emission: Vec3::zero(),
            max_distance: f64::INFINITY,
        }
    }

    /// E[pass weight] must equal the true transmittance, and scatter
    /// events integrate to sigma_s/sigma_t * (1 - T) (energy balance).
    #[test]
    fn homogeneous_estimator_unbiased() {
        let m = fog(0.6, 0.4); // sigma_t = 1
        let ray = Ray::new(Point3::origin(), Vec3::new(0.0, 0.0, 1.0));
        let t_max = 1.3;
        let mut rng = Pcg32::for_pixel_sample(3, 7);
        let n = 200_000;
        let mut pass = Vec3::zero();
        let mut scat = Vec3::zero();
        for _ in 0..n {
            match sample_distance(&m, &ray, t_max, &Vec3::one(), &mut rng) {
                MediumEvent::Pass { weight } => pass = pass + weight,
                MediumEvent::Scatter { weight, .. } => scat = scat + weight,
            }
        }
        let pass = pass / n as f64;
        let scat = scat / n as f64;
        let expected_pass = (-1.0f64 * t_max).exp();
        let expected_scat = 0.6 * (1.0 - expected_pass); // ∫ sigma_s T dt
        assert!((pass.x - expected_pass).abs() < 0.01, "pass {} vs {expected_pass}", pass.x);
        assert!((scat.x - expected_scat).abs() < 0.01, "scat {} vs {expected_scat}", scat.x);
    }

    /// Delta tracking must agree with the homogeneous closed form when the
    /// density field is constant 1 (fbm replaced by clamp -> use coverage
    /// 1, sharpness huge => density 1 everywhere noise > 0; instead test
    /// ratio-tracked transmittance against analytic on a real fbm field
    /// via a brute-force Riemann integral).
    #[test]
    fn ratio_tracking_matches_quadrature() {
        let m = Medium {
            sigma_a: Vec3::new(0.3, 0.3, 0.3),
            sigma_s: Vec3::new(0.7, 0.7, 0.7),
            g: 0.0,
            density: Some(DensityField::Fbm {
                params: crate::geometry::displace::DisplaceParams {
                    frequency: 0.8,
                    octaves: 3,
                    ..Default::default()
                },
                coverage: 0.6,
                sharpness: 3.0,
            }),
            emission: Vec3::zero(),
            max_distance: f64::INFINITY,
        };
        let origin = Point3::new(0.3, 0.2, -1.0);
        let dir = Vec3::new(0.2, 0.1, 1.0).normalize();
        let dist = 4.0;
        // Riemann optical depth.
        let steps = 20_000;
        let mut tau = 0.0;
        for i in 0..steps {
            let t = (i as f64 + 0.5) / steps as f64 * dist;
            tau += m.density_at(&(origin + dir * t)) * (dist / steps as f64);
        }
        let expected = (-1.0f64 * tau).exp(); // sigma_t = 1
        let mut rng = Pcg32::for_pixel_sample(11, 4);
        let n = 30_000;
        let mut sum = 0.0;
        for _ in 0..n {
            sum += transmittance(&m, &origin, &dir, dist, &mut rng).x;
        }
        let mean = sum / n as f64;
        assert!(
            (mean - expected).abs() < 0.02,
            "ratio tracking {mean} vs quadrature {expected}"
        );
    }
}
