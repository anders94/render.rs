//! Progressive Monte Carlo path tracer (Phase 1 of ROADMAP.md) — the CPU
//! reference integrator. Unidirectional path tracing with next-event
//! estimation, multiple importance sampling (power heuristic), and Russian
//! roulette. Materials map onto physically-based lobes: matte → Lambert,
//! plastic → Lambert + GGX coat, metal → GGX conductor.
//!
//! The legacy Whitted integrator remains the default until Phase 3
//! (`--integrator path` opts in); it is also what the GPU backends speak,
//! so image-parity tests stay pinned to it.

pub mod sampler;

use crate::math::{Point3, Vec3};
use crate::output::Image;
use crate::raytracer::{Intersection, Ray};
use crate::scene::{Light, LightType, Material, MaterialType, Scene};
use rayon::prelude::*;
use sampler::Pcg32;

const MAX_BOUNCES: usize = 8;
const RR_START: usize = 3;
/// Clamp per-sample luminance to tame fireflies (small bias, big variance win).
const FIREFLY_CLAMP: f64 = 60.0;
const RAY_OFFSET: f64 = 1e-4;

pub fn render(scene: &Scene, spp: u32) -> Image {
    let width = scene.camera.width as usize;
    let height = scene.camera.height as usize;

    (0..height)
        .into_par_iter()
        .map(|y| {
            (0..width)
                .map(|x| {
                    let pixel_index = (y * width + x) as u64;
                    let mut sum = Vec3::zero();
                    for s in 0..spp {
                        let mut rng = Pcg32::for_pixel_sample(pixel_index, s as u64);
                        let (jx, jy) = rng.next_2d();
                        let ray = scene
                            .camera
                            .generate_ray(x as f64 + jx, y as f64 + jy);
                        let mut l = trace(scene, ray, &mut rng);
                        let lum = luminance(&l);
                        if lum > FIREFLY_CLAMP {
                            l = l * (FIREFLY_CLAMP / lum);
                        }
                        sum = sum + l;
                    }
                    sum / spp as f64
                })
                .collect()
        })
        .collect()
}

fn luminance(c: &Vec3) -> f64 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

fn max_component(c: &Vec3) -> f64 {
    c.x.max(c.y).max(c.z)
}

fn trace(scene: &Scene, mut ray: Ray, rng: &mut Pcg32) -> Vec3 {
    let mut l = Vec3::zero();
    let mut beta = Vec3::one();
    // pdf of the previous BSDF sample (solid angle), for MIS on emitter hits.
    let mut prev_pdf = 0.0f64;
    let mut prev_origin = ray.origin;
    let mut from_camera = true;

    for depth in 0..MAX_BOUNCES {
        let Some(hit) = scene.intersect(&ray) else {
            l = l + beta * scene.background_color;
            break;
        };
        let material = &scene.materials[hit.material_id];
        let wo = -ray.direction.normalize();
        // Double-sided shading: normal faces the viewer.
        let n = if hit.normal.dot(&wo) < 0.0 { -hit.normal } else { hit.normal };

        // Emitter hit: full weight from the camera; MIS-weighted after a
        // BSDF bounce (NEE already sampled this light directly).
        if max_component(&material.emission) > 0.0 {
            let weight = if from_camera {
                1.0
            } else if let Some(light_idx) = material.area_light {
                let light = &scene.lights[light_idx];
                let pdf_light =
                    rect_pdf_solid_angle(light, &prev_origin, &hit) / scene.lights.len() as f64;
                power_heuristic(prev_pdf, pdf_light)
            } else {
                1.0
            };
            l = l + beta * material.emission * weight;
        }

        // Next-event estimation: sample one light uniformly.
        if !scene.lights.is_empty() {
            let light_idx = rng.next_below(scene.lights.len());
            let contribution =
                sample_light(scene, &scene.lights[light_idx], &hit.point, &n, &wo, material, rng);
            l = l + beta * contribution * scene.lights.len() as f64;
        }

        // Continue the path with a BSDF sample.
        let Some(bsdf_sample) = sample_bsdf(material, &wo, &n, rng) else {
            break;
        };
        let cos = bsdf_sample.wi.dot(&n).max(0.0);
        if bsdf_sample.pdf <= 0.0 || cos <= 0.0 {
            break;
        }
        beta = beta * bsdf_sample.f * (cos / bsdf_sample.pdf);
        prev_pdf = bsdf_sample.pdf;
        prev_origin = hit.point;
        from_camera = false;

        ray = Ray::new(hit.point + n * RAY_OFFSET, bsdf_sample.wi);

        // Russian roulette.
        if depth >= RR_START {
            let q = max_component(&beta).min(0.95);
            if q <= 0.0 || rng.next_f64() > q {
                break;
            }
            beta = beta / q;
        }
    }
    l
}

fn power_heuristic(pdf_a: f64, pdf_b: f64) -> f64 {
    let a2 = pdf_a * pdf_a;
    let b2 = pdf_b * pdf_b;
    if a2 + b2 <= 0.0 { 0.0 } else { a2 / (a2 + b2) }
}

/// Solid-angle pdf of a rect light generating the direction that produced
/// `hit`, as seen from `origin`.
fn rect_pdf_solid_angle(light: &Light, origin: &Point3, hit: &Intersection) -> f64 {
    let LightType::Rect { normal, area, .. } = &light.light_type else {
        return 0.0;
    };
    let d = hit.point - *origin;
    let dist2 = d.length_squared();
    let cos_light = (d.normalize().dot(normal)).abs();
    if cos_light < 1e-9 || *area <= 0.0 {
        return 0.0;
    }
    dist2 / (cos_light * area)
}

/// Direct lighting for one light (contribution already divided by the
/// light-selection pdf handled by the caller).
fn sample_light(
    scene: &Scene,
    light: &Light,
    p: &Point3,
    n: &Vec3,
    wo: &Vec3,
    material: &Material,
    rng: &mut Pcg32,
) -> Vec3 {
    match &light.light_type {
        LightType::Point { position } => {
            let to_light = *position - *p;
            let dist2 = to_light.length_squared().max(1e-12);
            let wi = to_light.normalize();
            let cos = wi.dot(n).max(0.0);
            if cos <= 0.0 || occluded_between(scene, p, n, position) {
                return Vec3::zero();
            }
            // Physically-based inverse-square falloff (the path tracer is
            // physical even where the Whitted integrator is not).
            bsdf_eval(material, wo, &wi, n) * light.radiance() * (cos / dist2)
        }
        LightType::Distant { direction } => {
            let wi = -*direction;
            let cos = wi.dot(n).max(0.0);
            if cos <= 0.0 || occluded_toward(scene, p, n, &wi) {
                return Vec3::zero();
            }
            bsdf_eval(material, wo, &wi, n) * light.radiance() * cos
        }
        LightType::Rect { corner, edge1, edge2, normal, area } => {
            let (u, v) = rng.next_2d();
            let sample_point = *corner + *edge1 * u + *edge2 * v;
            let to_light = sample_point - *p;
            let dist2 = to_light.length_squared().max(1e-12);
            let wi = to_light.normalize();
            let cos_surface = wi.dot(n).max(0.0);
            let cos_light = wi.dot(normal).abs(); // double-sided emitter
            if cos_surface <= 0.0 || cos_light < 1e-9 || *area <= 0.0 {
                return Vec3::zero();
            }
            if occluded_between(scene, p, n, &sample_point) {
                return Vec3::zero();
            }
            let pdf_sa = dist2 / (cos_light * area);
            let f = bsdf_eval(material, wo, &wi, n);
            let bsdf_pdf = bsdf_pdf(material, wo, &wi, n);
            let weight = power_heuristic(pdf_sa, bsdf_pdf);
            f * light.radiance() * (cos_surface / pdf_sa) * weight
        }
    }
}

fn occluded_between(scene: &Scene, p: &Point3, n: &Vec3, target: &Point3) -> bool {
    let dir = *target - *p;
    let dist = dir.length();
    let ray = Ray::new(*p + *n * RAY_OFFSET, dir.normalize());
    match scene.intersect(&ray) {
        Some(hit) => hit.t < dist - 1e-3,
        None => false,
    }
}

fn occluded_toward(scene: &Scene, p: &Point3, n: &Vec3, wi: &Vec3) -> bool {
    let ray = Ray::new(*p + *n * RAY_OFFSET, *wi);
    scene.intersect(&ray).is_some()
}

// ---------------------------------------------------------------------------
// BSDF lobes: matte → Lambert; plastic → Lambert + GGX (F0 = 0.04);
// metal → GGX conductor (F0 = base color). Roughness maps to alpha = r².

struct BsdfSample {
    wi: Vec3,
    f: Vec3,
    pdf: f64,
}

fn material_lobes(material: &Material) -> (Vec3, f64, f64) {
    // (diffuse albedo, ggx alpha, probability of sampling the specular lobe)
    match material.material_type {
        MaterialType::Matte => (material.color, 0.0, 0.0),
        MaterialType::Plastic { roughness } => {
            (material.color, alpha_from(roughness), 0.25)
        }
        MaterialType::Metal { roughness } => (Vec3::zero(), alpha_from(roughness), 1.0),
    }
}

fn alpha_from(roughness: f64) -> f64 {
    (roughness * roughness).clamp(1e-4, 1.0)
}

fn fresnel_schlick(f0: Vec3, cos: f64) -> Vec3 {
    let m = (1.0 - cos).clamp(0.0, 1.0).powi(5);
    f0 + (Vec3::one() - f0) * m
}

fn specular_f0(material: &Material) -> Vec3 {
    match material.material_type {
        MaterialType::Metal { .. } => material.color,
        _ => Vec3::new(0.04, 0.04, 0.04),
    }
}

fn ggx_d(n_dot_h: f64, alpha: f64) -> f64 {
    let a2 = alpha * alpha;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    a2 / (std::f64::consts::PI * d * d)
}

fn ggx_g1(n_dot_v: f64, alpha: f64) -> f64 {
    let a2 = alpha * alpha;
    let denom = n_dot_v + (a2 + (1.0 - a2) * n_dot_v * n_dot_v).sqrt();
    if denom <= 0.0 { 0.0 } else { 2.0 * n_dot_v / denom }
}

/// Build an orthonormal basis around n.
fn basis(n: &Vec3) -> (Vec3, Vec3) {
    let t = if n.x.abs() > 0.9 {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let b1 = n.cross(&t).normalize();
    let b2 = n.cross(&b1);
    (b1, b2)
}

fn cosine_sample(n: &Vec3, rng: &mut Pcg32) -> (Vec3, f64) {
    let (u, v) = rng.next_2d();
    let r = u.sqrt();
    let phi = 2.0 * std::f64::consts::PI * v;
    let (b1, b2) = basis(n);
    let wi = (b1 * (r * phi.cos()) + b2 * (r * phi.sin()) + *n * (1.0 - u).sqrt()).normalize();
    let pdf = wi.dot(n).max(0.0) / std::f64::consts::PI;
    (wi, pdf)
}

fn ggx_sample_half(n: &Vec3, alpha: f64, rng: &mut Pcg32) -> Vec3 {
    let (u, v) = rng.next_2d();
    let phi = 2.0 * std::f64::consts::PI * v;
    let cos_theta = ((1.0 - u) / (u * (alpha * alpha - 1.0) + 1.0)).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let (b1, b2) = basis(n);
    (b1 * (sin_theta * phi.cos()) + b2 * (sin_theta * phi.sin()) + *n * cos_theta).normalize()
}

fn bsdf_eval(material: &Material, wo: &Vec3, wi: &Vec3, n: &Vec3) -> Vec3 {
    let n_dot_o = n.dot(wo).max(0.0);
    let n_dot_i = n.dot(wi).max(0.0);
    if n_dot_o <= 0.0 || n_dot_i <= 0.0 {
        return Vec3::zero();
    }
    let (albedo, alpha, p_spec) = material_lobes(material);
    let mut f = Vec3::zero();
    if p_spec < 1.0 {
        f = f + albedo * (1.0 / std::f64::consts::PI);
    }
    if p_spec > 0.0 {
        let h = (*wo + *wi).normalize();
        let n_dot_h = n.dot(&h).max(0.0);
        let o_dot_h = wo.dot(&h).max(1e-9);
        let d = ggx_d(n_dot_h, alpha);
        let g = ggx_g1(n_dot_o, alpha) * ggx_g1(n_dot_i, alpha);
        let fresnel = fresnel_schlick(specular_f0(material), o_dot_h);
        f = f + fresnel * (d * g / (4.0 * n_dot_o * n_dot_i).max(1e-9));
    }
    f
}

fn bsdf_pdf(material: &Material, wo: &Vec3, wi: &Vec3, n: &Vec3) -> f64 {
    let n_dot_i = n.dot(wi).max(0.0);
    if n_dot_i <= 0.0 {
        return 0.0;
    }
    let (_, alpha, p_spec) = material_lobes(material);
    let pdf_diffuse = n_dot_i / std::f64::consts::PI;
    if p_spec <= 0.0 {
        return pdf_diffuse;
    }
    let h = (*wo + *wi).normalize();
    let n_dot_h = n.dot(&h).max(0.0);
    let o_dot_h = wo.dot(&h).abs().max(1e-9);
    let pdf_spec = ggx_d(n_dot_h, alpha) * n_dot_h / (4.0 * o_dot_h);
    (1.0 - p_spec) * pdf_diffuse + p_spec * pdf_spec
}

fn sample_bsdf(material: &Material, wo: &Vec3, n: &Vec3, rng: &mut Pcg32) -> Option<BsdfSample> {
    let (_, alpha, p_spec) = material_lobes(material);
    let wi = if rng.next_f64() < p_spec {
        let h = ggx_sample_half(n, alpha, rng);
        let wi = (-*wo).reflect(&h);
        if wi.dot(n) <= 0.0 {
            return None;
        }
        wi
    } else {
        cosine_sample(n, rng).0
    };
    let pdf = bsdf_pdf(material, wo, &wi, n);
    if pdf <= 0.0 {
        return None;
    }
    Some(BsdfSample {
        f: bsdf_eval(material, wo, &wi, n),
        wi,
        pdf,
    })
}
