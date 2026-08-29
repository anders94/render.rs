//! Progressive Monte Carlo path tracer — the CPU reference integrator.
//! Unidirectional path tracing with next-event estimation, multiple
//! importance sampling (power heuristic), and Russian roulette, over the
//! physically-based lobe system in `bxdf` (roadmap Phase 4): Oren-Nayar
//! diffuse, GGX/VNDF specular, clearcoat, fuzz, rough glass with true
//! refraction, glow, and presence cutouts. Lights: point, soft distant,
//! rect/sphere/disk area lights, and HDRI dome with importance sampling.

pub mod bxdf;
pub mod hair;
pub mod sampler;

use crate::math::{Point3, Vec3};
use crate::output::Image;
use crate::raytracer::{Intersection, Ray};
use crate::scene::{Light, LightType, Material, PbrParams, Scene};
use crate::texture::pattern::ShadeCtx;
use bxdf::Frame;
use rayon::prelude::*;
use sampler::Pcg32;
use std::f64::consts::PI;

const MAX_BOUNCES: usize = 10;
const RR_START: usize = 3;
/// Clamp per-sample luminance to tame fireflies (small bias, big variance win).
const FIREFLY_CLAMP: f64 = 60.0;
const RAY_OFFSET: f64 = 1e-4;
/// Presence pass-through events do not count as bounces (up to this many).
const MAX_PRESENCE_SKIPS: usize = 16;

pub fn render(scene: &Scene, spp: u32) -> Image {
    let width = scene.camera.width as usize;
    let height = scene.camera.height as usize;
    // Angular size of one pixel: the initial ray-cone spread for texture
    // footprints (task: mip selection without shimmer).
    let pixel_spread =
        (scene.camera.fov.to_radians() / 2.0).tan() * 2.0 / scene.camera.height as f64;

    (0..height)
        .into_par_iter()
        .map(|y| {
            (0..width)
                .map(|x| {
                    let pixel_index = (y * width + x) as u64;
                    let mut sum = Vec3::zero();
                    for s in 0..spp {
                        let mut rng = Pcg32::for_pixel_sample(pixel_index, s as u64);
                        let ray = camera_ray(scene, x, y, &mut rng);
                        let mut l = trace(scene, ray, pixel_spread, &mut rng);
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

/// Adaptive sampling (roadmap Phase 7): each pixel keeps sampling until
/// its 95% confidence interval, relative to its luminance, drops below
/// `tolerance` — or `max_spp` is reached. Convergence is checked every 16
/// samples after a 32-sample warmup (Welford online variance). Returns
/// the image and the average samples actually taken per pixel.
pub fn render_adaptive(scene: &Scene, max_spp: u32, tolerance: f64) -> (Image, f64) {
    let width = scene.camera.width as usize;
    let height = scene.camera.height as usize;
    let pixel_spread =
        (scene.camera.fov.to_radians() / 2.0).tan() * 2.0 / scene.camera.height as f64;
    const WARMUP: u32 = 32;
    const CHECK_EVERY: u32 = 16;

    let rows: Vec<(Vec<Vec3>, u64)> = (0..height)
        .into_par_iter()
        .map(|y| {
            let mut row = Vec::with_capacity(width);
            let mut taken = 0u64;
            for x in 0..width {
                let pixel_index = (y * width + x) as u64;
                let mut mean = Vec3::zero();
                let mut m2 = 0.0f64; // luminance variance accumulator
                let mut lum_mean = 0.0f64;
                let mut n = 0u32;
                while n < max_spp.max(1) {
                    let mut rng = Pcg32::for_pixel_sample(pixel_index, n as u64);
                    let ray = camera_ray(scene, x, y, &mut rng);
                    let mut l = trace(scene, ray, pixel_spread, &mut rng);
                    let lum = luminance(&l);
                    if lum > FIREFLY_CLAMP {
                        l = l * (FIREFLY_CLAMP / lum);
                    }
                    n += 1;
                    let lum = luminance(&l);
                    let delta = lum - lum_mean;
                    lum_mean += delta / n as f64;
                    m2 += delta * (lum - lum_mean);
                    mean = mean + (l - mean) / n as f64;
                    if n >= WARMUP && n % CHECK_EVERY == 0 {
                        let var = m2 / (n as f64 - 1.0);
                        let ci = 1.96 * (var / n as f64).sqrt();
                        if ci < tolerance * lum_mean.max(0.05) {
                            break;
                        }
                    }
                }
                taken += n as u64;
                row.push(mean);
            }
            (row, taken)
        })
        .collect();

    let total: u64 = rows.iter().map(|(_, t)| t).sum();
    let avg = total as f64 / (width * height) as f64;
    (rows.into_iter().map(|(row, _)| row).collect(), avg)
}

/// One camera ray: filter-importance-sampled subpixel position, thin-lens
/// aperture sample when DoF is on, shutter time when the scene moves.
fn camera_ray(scene: &Scene, x: usize, y: usize, rng: &mut Pcg32) -> Ray {
    let (u1, u2) = rng.next_2d();
    let (dx, dy) = scene.camera.filter.sample(u1, u2);
    let (px, py) = (x as f64 + 0.5 + dx, y as f64 + 0.5 + dy);
    let (lu, lv) = if scene.camera.lens_radius > 0.0 {
        rng.next_2d()
    } else {
        (0.5, 0.5)
    };
    let ray = scene.camera.generate_ray_lens(px, py, lu, lv);
    if scene.has_motion {
        ray.with_time(rng.next_f64())
    } else {
        ray
    }
}

fn luminance(c: &Vec3) -> f64 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

fn max_component(c: &Vec3) -> f64 {
    c.x.max(c.y).max(c.z)
}

/// Dome light lookup: (index, light) when the scene has one.
fn dome_of(scene: &Scene) -> Option<(usize, &Light)> {
    scene
        .lights
        .iter()
        .enumerate()
        .find(|(_, l)| matches!(l.light_type, LightType::Dome))
}

fn dome_radiance(light: &Light, dir: &Vec3) -> Vec3 {
    match &light.env {
        Some(env) => env.eval(dir) * light.radiance(),
        None => light.radiance(),
    }
}

fn dome_pdf(light: &Light, dir: &Vec3) -> f64 {
    match &light.env {
        Some(env) => env.pdf(dir),
        None => 1.0 / (4.0 * PI),
    }
}

/// Pattern-resolved lobe parameters at a hit (ray-cone footprint feeds
/// the texture mip selection).
fn shade_params(
    scene: &Scene,
    material: &Material,
    hit: &Intersection,
    cone_width: f64,
) -> PbrParams {
    if material.pattern_bindings.is_empty() {
        return material.pbr.clone();
    }
    let ctx = ShadeCtx {
        st: hit.st,
        p: [hit.point.x, hit.point.y, hit.point.z],
        n: [hit.normal.x, hit.normal.y, hit.normal.z],
        footprint: cone_width * hit.st_density,
    };
    material.resolved_pbr(&scene.patterns, &ctx)
}

fn trace(scene: &Scene, mut ray: Ray, pixel_spread: f64, rng: &mut Pcg32) -> Vec3 {
    let mut l = Vec3::zero();
    let mut beta = Vec3::one();
    // pdf of the previous BSDF sample (solid angle), for MIS on emitter hits.
    let mut prev_pdf = 0.0f64;
    let mut prev_origin = ray.origin;
    let mut from_camera = true;
    let mut presence_skips = 0usize;
    let num_lights = scene.lights.len() as f64;
    // Ray cone for texture filtering: width grows with distance, spread
    // widens after rough bounces.
    let mut cone_width = 0.0f64;
    let mut cone_spread = pixel_spread;

    let mut depth = 0usize;
    while depth < MAX_BOUNCES {
        let Some(hit) = scene.intersect(&ray) else {
            // Miss: dome light (with MIS) or flat background.
            if let Some((_, dome)) = dome_of(scene) {
                let dir = ray.direction.normalize();
                let weight = if from_camera {
                    1.0
                } else {
                    let pdf_light = dome_pdf(dome, &dir) / num_lights;
                    power_heuristic(prev_pdf, pdf_light)
                };
                l = l + beta * dome_radiance(dome, &dir) * weight;
            } else {
                l = l + beta * scene.background_color;
            }
            break;
        };
        let material = &scene.materials[hit.material_id];
        let wo = -ray.direction.normalize();
        cone_width += hit.t * cone_spread;
        let pbr = shade_params(scene, material, &hit, cone_width);

        // Presence cutout: stochastically pass through.
        let presence = pbr.presence.clamp(0.0, 1.0);
        if presence < 1.0 && rng.next_f64() >= presence && presence_skips < MAX_PRESENCE_SKIPS {
            presence_skips += 1;
            ray = Ray::new(hit.point + ray.direction.normalize() * RAY_OFFSET, ray.direction)
                .with_time(ray.time);
            continue;
        }

        // Geometric side decides the relative IOR for glass; shading uses
        // the viewer-facing normal.
        let entering = hit.front_face;
        let n = if hit.normal.dot(&wo) >= 0.0 { hit.normal } else { -hit.normal };
        let eta_rel = if entering {
            pbr.glass_ior
        } else {
            1.0 / pbr.glass_ior
        };

        // Emitter hit.
        let emission = material.emission + pbr.glow;
        if max_component(&emission) > 0.0 {
            let weight = if from_camera {
                1.0
            } else if let Some(light_idx) = material.area_light {
                let light = &scene.lights[light_idx];
                let pdf_light =
                    light_pdf_solid_angle(light, &prev_origin, &hit) / num_lights;
                power_heuristic(prev_pdf, pdf_light)
            } else {
                1.0
            };
            l = l + beta * emission * weight;
        }

        // Hair fibers scatter over the full sphere with their own BSDF;
        // they bypass the surface-lobe machinery entirely.
        if let Some(hp) = &material.hair {
            if hit.tangent.length_squared() > 0.5 {
                let ff = hair::FiberFrame::new(&hit.tangent, &hit.normal, &wo);
                let wo_f = ff.to_local(&wo);
                let hh = ff.h;
                if !scene.lights.is_empty() {
                    let eval = |wi_world: &Vec3| -> (Vec3, f64) {
                        let wi_f = ff.to_local(wi_world);
                        let fv = hair::f(hp, &wo_f, &wi_f, hh);
                        let pv = hair::pdf(hp, &wo_f, &wi_f, hh);
                        (fv, pv)
                    };
                    let light_idx = rng.next_below(scene.lights.len());
                    let c = sample_light(
                        scene,
                        &scene.lights[light_idx],
                        &hit.point,
                        n,
                        &eval,
                        ray.time,
                        rng,
                    );
                    l = l + beta * c * num_lights;
                }
                let Some((wi_f, fv, pv)) = hair::sample(hp, &wo_f, hh, rng) else {
                    break;
                };
                let wi_world = ff.to_world(&wi_f);
                beta = beta * fv / pv; // f is per-solid-angle: no cosine
                prev_pdf = pv;
                prev_origin = hit.point;
                from_camera = false;
                cone_spread += 0.4;
                let side = if wi_world.dot(&hit.normal) >= 0.0 { hit.normal } else { -hit.normal };
                ray = Ray::new(hit.point + side * RAY_OFFSET, wi_world).with_time(ray.time);
                depth += 1;
                if depth >= RR_START {
                    let q = max_component(&beta).min(0.95);
                    if q <= 0.0 || rng.next_f64() > q {
                        break;
                    }
                    beta = beta / q;
                }
                continue;
            }
        }

        let frame = Frame::new(n);
        let wo_l = frame.to_local(&wo);

        // Next-event estimation: sample one light uniformly.
        if !scene.lights.is_empty() && pbr.lobe_weights().is_some() {
            let eval = |wi_world: &Vec3| -> (Vec3, f64) {
                let wi_l = frame.to_local(wi_world);
                if wi_l.z <= 0.0 {
                    return (Vec3::zero(), 0.0);
                }
                let (f, pdf) = bxdf::eval_pdf(&pbr, &wo_l, &wi_l, eta_rel);
                (f * wi_l.z, pdf)
            };
            let light_idx = rng.next_below(scene.lights.len());
            let contribution = sample_light(
                scene,
                &scene.lights[light_idx],
                &hit.point,
                n,
                &eval,
                ray.time,
                rng,
            );
            l = l + beta * contribution * num_lights;
        }

        // Continue the path with a BSDF sample.
        let Some(s) = bxdf::sample(&pbr, &wo_l, eta_rel, rng) else {
            break;
        };
        let wi_world = frame.to_world(&s.wi);
        beta = beta * s.f * (s.wi.z.abs() / s.pdf);
        prev_pdf = s.pdf;
        prev_origin = hit.point;
        from_camera = false;
        // Widen the cone after rough bounces (tight lobes = high pdf add
        // almost nothing; diffuse adds ~0.3 rad).
        cone_spread += (1.0 / (1.0 + s.pdf)).min(0.4);

        let offset = if s.transmitted { -RAY_OFFSET } else { RAY_OFFSET };
        ray = Ray::new(hit.point + n * offset, wi_world).with_time(ray.time);
        depth += 1;

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

/// Solid-angle pdf of an area light generating the direction that produced
/// `hit`, as seen from `origin` (for MIS weighting of emitter hits).
fn light_pdf_solid_angle(light: &Light, origin: &Point3, hit: &Intersection) -> f64 {
    match &light.light_type {
        LightType::Rect { normal, area, .. } | LightType::DiskArea { normal, area, .. } => {
            let d = hit.point - *origin;
            let dist2 = d.length_squared();
            let cos_light = (d.normalize().dot(normal)).abs();
            if cos_light < 1e-9 || *area <= 0.0 {
                return 0.0;
            }
            dist2 / (cos_light * area)
        }
        LightType::SphereArea { center, radius } => {
            let dist2 = center.distance_squared(origin);
            let sin2 = (radius * radius / dist2).min(1.0);
            let cos_max = (1.0 - sin2).max(0.0).sqrt();
            let solid_angle = 2.0 * PI * (1.0 - cos_max);
            if solid_angle < 1e-12 {
                return 0.0;
            }
            1.0 / solid_angle
        }
        _ => 0.0,
    }
}

/// Direct lighting for one light. `eval` returns the BSDF value with any
/// cosine factor already applied (surfaces: f·cos⁺; hair: f alone) plus
/// the BSDF's solid-angle pdf for MIS. `n` is only the shadow-bias
/// direction.
#[allow(clippy::too_many_arguments)]
fn sample_light(
    scene: &Scene,
    light: &Light,
    p: &Point3,
    n: Vec3,
    eval: &dyn Fn(&Vec3) -> (Vec3, f64),
    time: f64,
    rng: &mut Pcg32,
) -> Vec3 {

    match &light.light_type {
        LightType::Point { position } => {
            let to_light = *position - *p;
            let dist2 = to_light.length_squared().max(1e-12);
            let wi = to_light.normalize();
            let (f, _) = eval(&wi);
            if max_component(&f) <= 0.0 {
                return Vec3::zero();
            }
            let vis = visibility_to(scene, p, &n, position, time);
            if vis <= 0.0 {
                return Vec3::zero();
            }
            f * light.radiance() * (vis / dist2)
        }
        LightType::Distant { direction, angular_radius } => {
            let base = -*direction;
            let (wi, pdf_sa) = if *angular_radius > 1e-5 {
                let cos_max = angular_radius.cos();
                let (u, v) = rng.next_2d();
                let cos_t = 1.0 - u * (1.0 - cos_max);
                let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
                let phi = 2.0 * PI * v;
                let lf = Frame::new(base);
                let wi = lf.to_world(&Vec3::new(
                    sin_t * phi.cos(),
                    sin_t * phi.sin(),
                    cos_t,
                ));
                (wi, 1.0 / (2.0 * PI * (1.0 - cos_max)))
            } else {
                (base, 0.0)
            };
            let (f, _) = eval(&wi);
            if max_component(&f) <= 0.0 {
                return Vec3::zero();
            }
            let vis = visibility_toward(scene, p, &n, &wi, time);
            if vis <= 0.0 {
                return Vec3::zero();
            }
            // Soft sun: radiance constant over the cone; the cone pdf
            // divides itself out.
            let _ = pdf_sa;
            f * light.radiance() * vis
        }
        LightType::Rect { corner, edge1, edge2, normal, area } => {
            let (u, v) = rng.next_2d();
            let sample_point = *corner + *edge1 * u + *edge2 * v;
            area_light_contribution(
                scene, p, &n, &sample_point, normal, *area, light, eval, time,
            )
        }
        LightType::DiskArea { center, e1, e2, normal, area } => {
            // Uniform disk sample.
            let (u, v) = rng.next_2d();
            let r = u.sqrt();
            let phi = 2.0 * PI * v;
            let sample_point = *center + *e1 * (r * phi.cos()) + *e2 * (r * phi.sin());
            area_light_contribution(
                scene, p, &n, &sample_point, normal, *area, light, eval, time,
            )
        }
        LightType::SphereArea { center, radius } => {
            let to_center = *center - *p;
            let dist2 = to_center.length_squared();
            if dist2 <= radius * radius * 1.0001 {
                return Vec3::zero(); // inside the light
            }
            let sin2 = (radius * radius / dist2).min(1.0);
            let cos_max = (1.0 - sin2).max(0.0).sqrt();
            let (u, v) = rng.next_2d();
            let cos_t = 1.0 - u * (1.0 - cos_max);
            let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
            let phi = 2.0 * PI * v;
            let lf = Frame::new(to_center.normalize());
            let wi = lf.to_world(&Vec3::new(sin_t * phi.cos(), sin_t * phi.sin(), cos_t));
            let (f, bsdf_pdf) = eval(&wi);
            if max_component(&f) <= 0.0 {
                return Vec3::zero();
            }
            // Occlusion up to the sphere surface along wi.
            let dist = dist2.sqrt() * cos_t
                - (radius * radius - dist2 * sin_t * sin_t).max(0.0).sqrt();
            let target = *p + wi * dist;
            let vis = visibility_to(scene, p, &n, &target, time);
            if vis <= 0.0 {
                return Vec3::zero();
            }
            let pdf_sa = 1.0 / (2.0 * PI * (1.0 - cos_max)).max(1e-12);
            let weight = power_heuristic(pdf_sa, bsdf_pdf);
            f * light.radiance() * (vis / pdf_sa) * weight
        }
        LightType::Dome => {
            let (wi, radiance, pdf_sa) = match &light.env {
                Some(env) => {
                    let (u, v) = rng.next_2d();
                    let (dir, rad, pdf) = env.sample(u, v);
                    (dir, rad * light.radiance(), pdf)
                }
                None => {
                    // Constant dome: uniform sphere, matching dome_pdf()
                    // exactly (the MIS weights must use the same density on
                    // both the NEE and BSDF-hit sides).
                    let (u, v) = rng.next_2d();
                    let z = 1.0 - 2.0 * u;
                    let r = (1.0 - z * z).max(0.0).sqrt();
                    let phi = 2.0 * PI * v;
                    let wi = Vec3::new(r * phi.cos(), r * phi.sin(), z);
                    (wi, light.radiance(), 1.0 / (4.0 * PI))
                }
            };
            if pdf_sa <= 0.0 {
                return Vec3::zero();
            }
            let (f, bsdf_pdf) = eval(&wi);
            if max_component(&f) <= 0.0 {
                return Vec3::zero();
            }
            let vis = visibility_toward(scene, p, &n, &wi, time);
            if vis <= 0.0 {
                return Vec3::zero();
            }
            let weight = power_heuristic(pdf_sa, bsdf_pdf);
            f * radiance * (vis / pdf_sa) * weight
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn area_light_contribution(
    scene: &Scene,
    p: &Point3,
    n: &Vec3,
    sample_point: &Point3,
    light_normal: &Vec3,
    area: f64,
    light: &Light,
    eval: &dyn Fn(&Vec3) -> (Vec3, f64),
    time: f64,
) -> Vec3 {
    let to_light = *sample_point - *p;
    let dist2 = to_light.length_squared().max(1e-12);
    let wi = to_light.normalize();
    let cos_light = wi.dot(light_normal).abs(); // double-sided emitter
    if cos_light < 1e-9 || area <= 0.0 {
        return Vec3::zero();
    }
    let (f, bsdf_pdf) = eval(&wi);
    if max_component(&f) <= 0.0 {
        return Vec3::zero();
    }
    let vis = visibility_to(scene, p, n, sample_point, time);
    if vis <= 0.0 {
        return Vec3::zero();
    }
    let pdf_sa = dist2 / (cos_light * area);
    let weight = power_heuristic(pdf_sa, bsdf_pdf);
    f * light.radiance() * (vis / pdf_sa) * weight
}

/// Fractional visibility toward a point (presence cutouts attenuate).
fn visibility_to(scene: &Scene, p: &Point3, n: &Vec3, target: &Point3, time: f64) -> f64 {
    let dir = *target - *p;
    let dist = dir.length();
    transmittance(scene, *p + *n * RAY_OFFSET, dir.normalize(), dist - 1e-3, time)
}

fn visibility_toward(scene: &Scene, p: &Point3, n: &Vec3, wi: &Vec3, time: f64) -> f64 {
    transmittance(scene, *p + *n * RAY_OFFSET, *wi, f64::INFINITY, time)
}

/// Walks blockers along a shadow ray; opaque surfaces kill it, presence
/// cutouts multiply through.
fn transmittance(
    scene: &Scene,
    mut origin: Point3,
    dir: Vec3,
    mut remaining: f64,
    time: f64,
) -> f64 {
    let mut vis = 1.0;
    for _ in 0..MAX_PRESENCE_SKIPS {
        let ray = Ray::new(origin, dir).with_time(time);
        let Some(hit) = scene.intersect(&ray) else {
            return vis;
        };
        if hit.t >= remaining {
            return vis;
        }
        let material = &scene.materials[hit.material_id];
        let presence = shade_params(scene, material, &hit, 0.0).presence.clamp(0.0, 1.0);
        if presence >= 1.0 {
            return 0.0;
        }
        vis *= 1.0 - presence;
        if vis < 1e-4 {
            return 0.0;
        }
        origin = hit.point + dir * RAY_OFFSET;
        remaining -= hit.t + RAY_OFFSET;
    }
    vis
}
