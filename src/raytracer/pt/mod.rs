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
pub mod volume;

use crate::math::{Point3, Vec3};
use crate::output::film::{AuxSample, Film};
use crate::output::Image;
use crate::raytracer::{Intersection, Ray};
use crate::scene::medium::{hg_phase, hg_sample};
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

/// Nested-media bookkeeping: a small stack of medium indices; the top is
/// the medium the ray currently travels in, the atmosphere is the
/// implicit bottom. Entering a hull/glass interior pushes, exiting pops —
/// so a glass ball inside a cloud correctly returns rays to the CLOUD,
/// not to clear air (the old single-slot tracking got this wrong).
#[derive(Clone, Copy)]
struct MediumStack {
    stack: [u32; 8],
    depth: usize,
    atmosphere: Option<u32>,
}

impl MediumStack {
    fn new(atmosphere: Option<u32>) -> Self {
        Self { stack: [0; 8], depth: 0, atmosphere }
    }

    fn current(&self) -> Option<u32> {
        if self.depth > 0 {
            Some(self.stack[self.depth - 1])
        } else {
            self.atmosphere
        }
    }

    fn push(&mut self, medium: u32) {
        if self.depth < 8 {
            self.stack[self.depth] = medium;
            self.depth += 1;
        }
    }

    fn pop(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }
}

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

/// One progressive sample pass: sample index `sample` for every pixel
/// (deterministic per-(pixel, sample) seeding makes accumulation across
/// calls equal a batch render). The interactive preview drives this.
pub fn render_one(scene: &Scene, sample: u32) -> Image {
    let width = scene.camera.width as usize;
    let height = scene.camera.height as usize;
    let pixel_spread =
        (scene.camera.fov.to_radians() / 2.0).tan() * 2.0 / scene.camera.height as f64;
    (0..height)
        .into_par_iter()
        .map(|y| {
            (0..width)
                .map(|x| {
                    let pixel_index = (y * width + x) as u64;
                    let mut rng = Pcg32::for_pixel_sample(pixel_index, sample as u64);
                    let ray = camera_ray(scene, x, y, &mut rng);
                    let mut l = trace(scene, ray, pixel_spread, &mut rng);
                    let lum = luminance(&l);
                    if lum > FIREFLY_CLAMP {
                        l = l * (FIREFLY_CLAMP / lum);
                    }
                    l
                })
                .collect()
        })
        .collect()
}

/// Render with the full AOV stack (roadmap Phase 11): beauty plus
/// diffuse/specular split, first-hit albedo/normal/depth, and the
/// dominant-object id layer with coverage.
pub fn render_film(scene: &Scene, spp: u32) -> Film {
    let width = scene.camera.width as usize;
    let height = scene.camera.height as usize;
    let pixel_spread =
        (scene.camera.fov.to_radians() / 2.0).tan() * 2.0 / scene.camera.height as f64;

    struct Row {
        beauty: Vec<Vec3>,
        diffuse: Vec<Vec3>,
        albedo: Vec<Vec3>,
        normal: Vec<Vec3>,
        depth: Vec<Vec3>,
        id: Vec<Vec3>,
    }
    let rows: Vec<Row> = (0..height)
        .into_par_iter()
        .map(|y| {
            let mut row = Row {
                beauty: Vec::with_capacity(width),
                diffuse: Vec::with_capacity(width),
                albedo: Vec::with_capacity(width),
                normal: Vec::with_capacity(width),
                depth: Vec::with_capacity(width),
                id: Vec::with_capacity(width),
            };
            for x in 0..width {
                let pixel_index = (y * width + x) as u64;
                let mut sum = Vec3::zero();
                let mut d_sum = Vec3::zero();
                let mut alb = Vec3::zero();
                let mut nrm = Vec3::zero();
                let mut depth_sum = 0.0f64;
                let mut hits = 0u32;
                let mut votes: std::collections::HashMap<u32, u32> =
                    std::collections::HashMap::new();
                for s in 0..spp {
                    let mut rng = Pcg32::for_pixel_sample(pixel_index, s as u64);
                    let ray = camera_ray(scene, x, y, &mut rng);
                    let mut aux = AuxSample::default();
                    let mut l = trace_full(scene, ray, pixel_spread, &mut rng, &mut aux);
                    let lum = luminance(&l);
                    if lum > FIREFLY_CLAMP {
                        let k = FIREFLY_CLAMP / lum;
                        l = l * k;
                        aux.diffuse = aux.diffuse * k;
                    }
                    sum = sum + l;
                    d_sum = d_sum + aux.diffuse;
                    alb = alb + aux.albedo;
                    nrm = nrm + aux.normal;
                    if aux.depth > 0.0 {
                        depth_sum += aux.depth;
                        hits += 1;
                    }
                    *votes.entry(aux.id).or_insert(0) += 1;
                }
                let inv = 1.0 / spp as f64;
                let (best_id, best_n) = votes
                    .into_iter()
                    .max_by_key(|(_, n)| *n)
                    .unwrap_or((0, 0));
                row.beauty.push(sum * inv);
                row.diffuse.push(d_sum * inv);
                row.albedo.push(alb * inv);
                row.normal.push(nrm * inv);
                row.depth.push(Vec3::new(
                    if hits > 0 { depth_sum / hits as f64 } else { 0.0 },
                    0.0,
                    0.0,
                ));
                row.id.push(Vec3::new(
                    best_id as f64,
                    best_n as f64 * inv,
                    0.0,
                ));
            }
            row
        })
        .collect();

    let mut film = Film {
        beauty: Vec::with_capacity(height),
        diffuse: Vec::with_capacity(height),
        specular: Vec::with_capacity(height),
        albedo: Vec::with_capacity(height),
        normal: Vec::with_capacity(height),
        depth: Vec::with_capacity(height),
        id: Vec::with_capacity(height),
        manifest: scene.id_manifest.clone(),
    };
    for row in rows {
        let spec: Vec<Vec3> = row
            .beauty
            .iter()
            .zip(&row.diffuse)
            .map(|(b, d)| {
                Vec3::new(
                    (b.x - d.x).max(0.0),
                    (b.y - d.y).max(0.0),
                    (b.z - d.z).max(0.0),
                )
            })
            .collect();
        film.beauty.push(row.beauty);
        film.diffuse.push(row.diffuse);
        film.specular.push(spec);
        film.albedo.push(row.albedo);
        film.normal.push(row.normal);
        film.depth.push(row.depth);
        film.id.push(row.id);
    }
    film
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
        let t = rng.next_f64();
        scene.camera.apply_motion(ray.with_time(t), t)
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
    ray_dir: &Vec3,
    cone_width: f64,
) -> PbrParams {
    if material.pattern_bindings.is_empty() {
        return material.pbr.clone();
    }
    // Anisotropic footprint (EWA): the ray cone's disk projects onto the
    // surface as an ellipse — minor axis = cone width, major axis
    // stretched by 1/|cos θ| along the projected view direction. The
    // st Jacobian turns both into texture-space axes.
    let (dst_major, dst_minor) = {
        let gs = hit.st_grad[0];
        let gt = hit.st_grad[1];
        if gs.length_squared() + gt.length_squared() > 1e-18 && cone_width > 0.0 {
            let n = hit.normal;
            let d = ray_dir.normalize();
            let cos = d.dot(&n).abs().max(1.0 / 16.0);
            let mut t1 = d - n * d.dot(&n);
            if t1.length_squared() < 1e-12 {
                t1 = if n.x.abs() < 0.9 {
                    Vec3::new(1.0, 0.0, 0.0).cross(&n)
                } else {
                    Vec3::new(0.0, 1.0, 0.0).cross(&n)
                };
            }
            let t1 = t1.normalize() * (cone_width / cos);
            let t2 = n.cross(&t1).normalize() * cone_width;
            (
                [gs.dot(&t1), gt.dot(&t1)],
                [gs.dot(&t2), gt.dot(&t2)],
            )
        } else {
            ([0.0, 0.0], [0.0, 0.0])
        }
    };
    let ctx = ShadeCtx {
        st: hit.st,
        p: [hit.point.x, hit.point.y, hit.point.z],
        n: [hit.normal.x, hit.normal.y, hit.normal.z],
        footprint: cone_width * hit.st_density,
        dst_major,
        dst_minor,
    };
    material.resolved_pbr(&scene.patterns, &ctx)
}

fn trace(scene: &Scene, ray: Ray, pixel_spread: f64, rng: &mut Pcg32) -> Vec3 {
    let mut aux = AuxSample::default();
    trace_full(scene, ray, pixel_spread, rng, &mut aux)
}

/// Full trace: beauty radiance plus auxiliary AOV data (first-hit
/// albedo/normal/depth/id and the diffuse share of the beauty — the
/// specular layer is beauty minus diffuse).
fn trace_full(
    scene: &Scene,
    mut ray: Ray,
    pixel_spread: f64,
    rng: &mut Pcg32,
    aux: &mut AuxSample,
) -> Vec3 {
    let mut l = Vec3::zero();
    // Which lobe the FIRST path continuation used (None until sampled);
    // contributions land in the diffuse AOV unless it was specular.
    let mut first_specular: Option<bool> = None;
    let mut aux_set = false;
    let mut travel = 0.0f64;
    macro_rules! emit {
        ($c:expr) => {{
            let c = $c;
            l = l + c;
            if first_specular != Some(true) {
                aux.diffuse = aux.diffuse + c;
            }
        }};
    }
    let mut beta = Vec3::one();
    // pdf of the previous BSDF sample (solid angle), for MIS on emitter hits.
    let mut prev_pdf = 0.0f64;
    let mut prev_origin = ray.origin;
    let mut from_camera = true;
    let mut presence_skips = 0usize;
    // Ray cone for texture filtering: width grows with distance, spread
    // widens after rough bounces.
    let mut cone_width = 0.0f64;
    let mut cone_spread = pixel_spread;
    // Participating media the ray is inside of (nested).
    let mut med_stack = MediumStack::new(scene.atmosphere);
    // Multiple scattering in clouds needs a deeper budget than surfaces.
    let max_bounces = if scene.media.is_empty() { MAX_BOUNCES } else { 64 };

    let mut depth = 0usize;
    while depth < max_bounces {
        let hit_opt = scene.intersect(&ray);

        // Medium interaction along this segment (before any surface).
        let medium = med_stack.current();
        if let Some(mid) = medium {
            let med = &scene.media[mid as usize];
            let t_limit = hit_opt.as_ref().map(|h| h.t).unwrap_or(1e8);

            // One light pick per segment, shared by both direct-lighting
            // strategies (equiangular + collision NEE) so their MIS
            // weights cancel the selection pmf exactly.
            let seg_pick = scene
                .light_sampler
                .sample(&ray.at((t_limit.min(med.max_distance) * 0.5).min(50.0)), rng.next_f64());

            // Equiangular strategy: for point lights, place a dedicated
            // single-scatter NEE vertex where the 1/r² glow concentrates.
            if let Some((li, pick_pmf)) = seg_pick {
                if let LightType::Point { position } = &scene.lights[li].light_type {
                    let t_seg = t_limit.min(med.max_distance);
                    if t_seg > 1e-6 {
                        let (t_eq, pdf_eq) = volume::equiangular_sample(
                            &ray.origin,
                            &ray.direction,
                            t_seg,
                            position,
                            rng.next_f64(),
                        );
                        if pdf_eq > 1e-12 {
                            let p = ray.at(t_eq);
                            let rho = med.density_at(&p);
                            if rho > 0.0 {
                                let tr = volume::transmittance(
                                    med, &ray.origin, &ray.direction, t_eq, rng,
                                );
                                let sigma_s = med.sigma_s * rho;
                                let to_l = *position - p;
                                let dist2 = to_l.length_squared().max(1e-12);
                                let wi = to_l.normalize();
                                let wo = -ray.direction.normalize();
                                let ph = hg_phase(med.g, -wo.dot(&wi));
                                let vis = transmittance(
                                    scene,
                                    p,
                                    wi,
                                    dist2.sqrt() - 1e-3,
                                    ray.time,
                                    medium,
                                    rng,
                                );
                                if max_component(&vis) > 0.0 {
                                    let pdf_d =
                                        volume::distance_pdf_approx(med, t_eq, t_seg);
                                    let w_eq = pdf_eq / (pdf_eq + pdf_d);
                                    let c = sigma_s
                                        * tr
                                        * vis
                                        * scene.lights[li].radiance()
                                        * (ph * w_eq / (dist2 * pdf_eq * pick_pmf));
                                    emit!(beta * c);
                                }
                            }
                        }
                    }
                }
            }

            match volume::sample_distance(med, &ray, t_limit, &beta, rng) {
                volume::MediumEvent::Scatter { t, weight } => {
                    beta = beta * weight;
                    let p = ray.at(t);
                    if !aux_set {
                        aux_set = true;
                        let st = med.sigma_t();
                        aux.albedo = Vec3::new(
                            med.sigma_s.x / st.x.max(1e-9),
                            med.sigma_s.y / st.y.max(1e-9),
                            med.sigma_s.z / st.z.max(1e-9),
                        );
                        aux.normal = -ray.direction.normalize();
                        aux.depth = travel + t;
                        aux.id = 0;
                    }
                    if first_specular.is_none() {
                        first_specular = Some(false);
                    }
                    if max_component(&med.emission) > 0.0 {
                        emit!(beta * med.emission);
                    }
                    // NEE from the collision point, using the segment's
                    // shared light pick. Point lights carry the MIS weight
                    // pairing this strategy with equiangular; every other
                    // light type has no equiangular partner (weight 1).
                    let wo = -ray.direction.normalize();
                    let g = med.g;
                    if let Some((light_idx, pmf)) = seg_pick {
                        let w_dist = if let LightType::Point { position } =
                            &scene.lights[light_idx].light_type
                        {
                            let t_seg = t_limit.min(med.max_distance);
                            let pdf_d = volume::distance_pdf_approx(med, t, t_seg);
                            let pdf_e = volume::equiangular_pdf(
                                &ray.origin,
                                &ray.direction,
                                t_seg,
                                position,
                                t,
                            );
                            pdf_d / (pdf_d + pdf_e)
                        } else {
                            1.0
                        };
                        let eval = |wi: &Vec3| -> (Vec3, f64) {
                            let ph = hg_phase(g, -wo.dot(wi));
                            (Vec3::new(ph, ph, ph), ph)
                        };
                        let c = sample_light(
                            scene,
                            &scene.lights[light_idx],
                            &p,
                            Vec3::zero(),
                            &eval,
                            pmf,
                            ray.time,
                            medium,
                            rng,
                        );
                        emit!(beta * c * w_dist);
                    }
                    // Continue with an HG sample: phase/pdf = 1.
                    let (wi, pdf) = hg_sample(g, &wo, rng.next_f64(), rng.next_f64());
                    prev_pdf = pdf;
                    prev_origin = p;
                    from_camera = false;
                    cone_spread += 0.4;
                    ray = Ray::new(p, wi).with_time(ray.time);
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
                volume::MediumEvent::Pass { weight } => {
                    beta = beta * weight;
                    if max_component(&beta) <= 0.0 {
                        break;
                    }
                }
            }
        }

        let Some(hit) = hit_opt else {
            // Miss: dome light (with MIS) or flat background.
            if let Some((dome_idx, dome)) = dome_of(scene) {
                let dir = ray.direction.normalize();
                let weight = if from_camera {
                    1.0
                } else {
                    let pick = scene.light_sampler.pmf(&prev_origin, dome_idx);
                    let pdf_light = dome_pdf(dome, &dir) * pick;
                    power_heuristic(prev_pdf, pdf_light)
                };
                emit!(beta * dome_radiance(dome, &dir) * weight);
            } else {
                emit!(beta * scene.background_color);
            }
            break;
        };
        let material = &scene.materials[hit.material_id];

        // Invisible volume hull: no lobes, no emission, an interior medium
        // — the crossing just toggles the medium.
        if material.interior.is_some()
            && material.hair.is_none()
            && material.pbr.lobe_weights().is_none()
            && max_component(&(material.emission + material.pbr.glow)) <= 0.0
        {
            if presence_skips < MAX_PRESENCE_SKIPS {
                presence_skips += 1;
                travel += hit.t;
                if hit.front_face {
                    med_stack.push(material.interior.unwrap());
                } else {
                    med_stack.pop();
                }
                ray = Ray::new(
                    hit.point + ray.direction.normalize() * RAY_OFFSET,
                    ray.direction,
                )
                .with_time(ray.time);
                continue;
            }
        }

        let wo = -ray.direction.normalize();
        cone_width += hit.t * cone_spread;
        travel += hit.t;
        let pbr = shade_params(scene, material, &hit, &ray.direction, cone_width);
        if !aux_set {
            aux_set = true;
            let a = pbr.diffuse_color * pbr.diffuse_gain
                + pbr.specular_f0() * if pbr.has_specular() { 1.0 } else { 0.0 }
                + pbr.subsurface_color * pbr.subsurface_gain
                + pbr.refraction_color * pbr.glass_gain;
            aux.albedo = Vec3::new(a.x.min(1.0), a.y.min(1.0), a.z.min(1.0));
            aux.normal = if hit.normal.dot(&wo) >= 0.0 { hit.normal } else { -hit.normal };
            aux.depth = travel;
            aux.id = material.id;
        }

        // Presence cutout: stochastically pass through.
        let presence = pbr.presence.clamp(0.0, 1.0);
        if presence < 1.0 && rng.next_f64() >= presence && presence_skips < MAX_PRESENCE_SKIPS {
            presence_skips += 1;
            travel += hit.t;
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
                let pick = scene.light_sampler.pmf(&prev_origin, light_idx);
                let pdf_light = light_pdf_solid_angle(light, &prev_origin, &hit) * pick;
                power_heuristic(prev_pdf, pdf_light)
            } else {
                1.0
            };
            emit!(beta * emission * weight);
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
                    if let Some((light_idx, pmf)) =
                        scene.light_sampler.sample(&hit.point, rng.next_f64())
                    {
                        let c = sample_light(
                            scene,
                            &scene.lights[light_idx],
                            &hit.point,
                            n,
                            &eval,
                            pmf,
                            ray.time,
                            medium,
                            rng,
                        );
                        emit!(beta * c);
                    }
                }
                let Some((wi_f, fv, pv)) = hair::sample(hp, &wo_f, hh, rng) else {
                    break;
                };
                let wi_world = ff.to_world(&wi_f);
                beta = beta * fv / pv; // f is per-solid-angle: no cosine
                if first_specular.is_none() {
                    first_specular = Some(false);
                }
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

        // Subsurface scattering: with probability subsurfaceGain the path
        // enters an isotropic random walk inside the object (a homogeneous
        // medium derived from the Burley scaling fit), exiting diffusely
        // at the first surface the walk reaches.
        if pbr.subsurface_gain > 0.0
            && material.hair.is_none()
            && rng.next_f64() < pbr.subsurface_gain.clamp(0.0, 1.0)
        {
            if first_specular.is_none() {
                first_specular = Some(false);
            }
            let sss = volume::sss_medium(&pbr.subsurface_color, &pbr.subsurface_dmfp);
            // Diffuse transmission entry: cosine hemisphere around -n.
            let entry_frame = Frame::new(-n);
            let (u1, u2) = rng.next_2d();
            let r = u1.sqrt();
            let phi = 2.0 * PI * u2;
            let local = Vec3::new(
                r * phi.cos(),
                r * phi.sin(),
                (1.0 - u1).max(0.0).sqrt(),
            );
            let mut walk = Ray::new(hit.point - n * RAY_OFFSET, entry_frame.to_world(&local))
                .with_time(ray.time);
            let mut exited = false;
            for step in 0..256 {
                let Some(whit) = scene.intersect(&walk) else {
                    break; // escaped through a crack: kill the path
                };
                match volume::sample_distance(&sss, &walk, whit.t, &beta, rng) {
                    volume::MediumEvent::Scatter { t, weight } => {
                        // weight = sigma_s * T / pdf — the complete
                        // throughput multiplier for one walk step.
                        beta = beta * weight;
                        let p = walk.at(t);
                        let dir = {
                            let u = rng.next_f64();
                            let v = rng.next_f64();
                            let z = 1.0 - 2.0 * u;
                            let rr = (1.0 - z * z).max(0.0).sqrt();
                            let ph = 2.0 * PI * v;
                            Vec3::new(rr * ph.cos(), rr * ph.sin(), z)
                        };
                        // Isotropic phase: value 1/4π, pdf 1/4π — cancels.
                        walk = Ray::new(p, dir).with_time(ray.time);
                        let _ = step;
                        // Russian roulette inside long walks.
                        let q = max_component(&beta).min(0.95);
                        if q <= 0.0 || (step > 8 && rng.next_f64() > q) {
                            break;
                        }
                        if step > 8 {
                            beta = beta / q;
                        }
                    }
                    volume::MediumEvent::Pass { weight } => {
                        beta = beta * weight;
                        // Exit: diffuse (cosine) leave at the walk's surface
                        // hit, on the outside. Normal conventions differ by
                        // primitive (quadrics report outward, meshes flip
                        // toward the viewer) — outward is wherever the walk
                        // is heading.
                        let out_n = if whit.normal.dot(&walk.direction) > 0.0 {
                            whit.normal
                        } else {
                            -whit.normal
                        };
                        let exit_frame = Frame::new(out_n);
                        let exit_p = walk.at(whit.t);
                        // NEE at the exit vertex (white Lambert transmission).
                        if let Some((light_idx, pmf)) =
                            scene.light_sampler.sample(&exit_p, rng.next_f64())
                        {
                            let eval = |wi_world: &Vec3| -> (Vec3, f64) {
                                let wi_l = exit_frame.to_local(wi_world);
                                if wi_l.z <= 0.0 {
                                    return (Vec3::zero(), 0.0);
                                }
                                let f = wi_l.z / PI;
                                (Vec3::new(f, f, f), wi_l.z / PI)
                            };
                            let c = sample_light(
                                scene,
                                &scene.lights[light_idx],
                                &exit_p,
                                out_n,
                                &eval,
                                pmf,
                                ray.time,
                                medium,
                                rng,
                            );
                            emit!(beta * c);
                        }
                        let (u1, u2) = rng.next_2d();
                        let r = u1.sqrt();
                        let phi = 2.0 * PI * u2;
                        let local = Vec3::new(
                            r * phi.cos(),
                            r * phi.sin(),
                            (1.0 - u1).max(0.0).sqrt(),
                        );
                        let wi = exit_frame.to_world(&local);
                        prev_pdf = local.z.max(1e-9) / PI;
                        prev_origin = exit_p;
                        from_camera = false;
                        cone_spread += 0.4;
                        ray = Ray::new(exit_p + out_n * RAY_OFFSET, wi).with_time(ray.time);
                        exited = true;
                        break;
                    }
                }
            }
            if !exited {
                break;
            }
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
            if let Some((light_idx, pmf)) =
                scene.light_sampler.sample(&hit.point, rng.next_f64())
            {
                let contribution = sample_light(
                    scene,
                    &scene.lights[light_idx],
                    &hit.point,
                    n,
                    &eval,
                    pmf,
                    ray.time,
                    medium,
                    rng,
                );
                emit!(beta * contribution);
            }
        }

        // Continue the path with a BSDF sample.
        let Some(s) = bxdf::sample(&pbr, &wo_l, eta_rel, rng) else {
            break;
        };
        let wi_world = frame.to_world(&s.wi);
        beta = beta * s.f * (s.wi.z.abs() / s.pdf);
        if first_specular.is_none() {
            first_specular = Some(s.specular_lobe);
        }
        prev_pdf = s.pdf;
        prev_origin = hit.point;
        from_camera = false;
        // Widen the cone after rough bounces (tight lobes = high pdf add
        // almost nothing; diffuse adds ~0.3 rad).
        cone_spread += (1.0 / (1.0 + s.pdf)).min(0.4);

        if s.transmitted && material.interior.is_some() {
            if entering {
                med_stack.push(material.interior.unwrap());
            } else {
                med_stack.pop();
            }
        }
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
/// direction. `pick_pmf` is the light-selection probability: the full
/// light-strategy density is pick_pmf * pdf_sa, and BOTH the estimator
/// divisor and the MIS weight must use it (the emitter-hit side does).
#[allow(clippy::too_many_arguments)]
fn sample_light(
    scene: &Scene,
    light: &Light,
    p: &Point3,
    n: Vec3,
    eval: &dyn Fn(&Vec3) -> (Vec3, f64),
    pick_pmf: f64,
    time: f64,
    medium: Option<u32>,
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
            let vis = visibility_to(scene, p, &n, position, time, medium, rng);
            if max_component(&vis) <= 0.0 {
                return Vec3::zero();
            }
            f * light.radiance() * vis / (dist2 * pick_pmf)
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
            let vis = visibility_toward(scene, p, &n, &wi, time, medium, rng);
            if max_component(&vis) <= 0.0 {
                return Vec3::zero();
            }
            // Soft sun: radiance constant over the cone; the cone pdf
            // divides itself out. Delta/cone light: no BSDF-side MIS.
            let _ = pdf_sa;
            f * light.radiance() * vis / pick_pmf
        }
        LightType::Rect { corner, edge1, edge2, normal, area } => {
            let (u, v) = rng.next_2d();
            let sample_point = *corner + *edge1 * u + *edge2 * v;
            area_light_contribution(
                scene, p, &n, &sample_point, normal, *area, light, eval, pick_pmf, time,
                medium, rng,
            )
        }
        LightType::DiskArea { center, e1, e2, normal, area } => {
            // Uniform disk sample.
            let (u, v) = rng.next_2d();
            let r = u.sqrt();
            let phi = 2.0 * PI * v;
            let sample_point = *center + *e1 * (r * phi.cos()) + *e2 * (r * phi.sin());
            area_light_contribution(
                scene, p, &n, &sample_point, normal, *area, light, eval, pick_pmf, time,
                medium, rng,
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
            let vis = visibility_to(scene, p, &n, &target, time, medium, rng);
            if max_component(&vis) <= 0.0 {
                return Vec3::zero();
            }
            let pdf_sa = pick_pmf / (2.0 * PI * (1.0 - cos_max)).max(1e-12);
            let weight = power_heuristic(pdf_sa, bsdf_pdf);
            f * light.radiance() * vis * (weight / pdf_sa)
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
            let pdf_sa = pdf_sa * pick_pmf;
            let (f, bsdf_pdf) = eval(&wi);
            if max_component(&f) <= 0.0 {
                return Vec3::zero();
            }
            let vis = visibility_toward(scene, p, &n, &wi, time, medium, rng);
            if max_component(&vis) <= 0.0 {
                return Vec3::zero();
            }
            let weight = power_heuristic(pdf_sa, bsdf_pdf);
            f * radiance * vis * (weight / pdf_sa)
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
    pick_pmf: f64,
    time: f64,
    medium: Option<u32>,
    rng: &mut Pcg32,
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
    let vis = visibility_to(scene, p, n, sample_point, time, medium, rng);
    if max_component(&vis) <= 0.0 {
        return Vec3::zero();
    }
    let pdf_sa = pick_pmf * dist2 / (cos_light * area);
    let weight = power_heuristic(pdf_sa, bsdf_pdf);
    f * light.radiance() * vis * (weight / pdf_sa)
}

/// Fractional visibility toward a point (presence cutouts attenuate).
fn visibility_to(
    scene: &Scene,
    p: &Point3,
    n: &Vec3,
    target: &Point3,
    time: f64,
    medium: Option<u32>,
    rng: &mut Pcg32,
) -> Vec3 {
    let dir = *target - *p;
    let dist = dir.length();
    transmittance(scene, *p + *n * RAY_OFFSET, dir.normalize(), dist - 1e-3, time, medium, rng)
}

fn visibility_toward(
    scene: &Scene,
    p: &Point3,
    n: &Vec3,
    wi: &Vec3,
    time: f64,
    medium: Option<u32>,
    rng: &mut Pcg32,
) -> Vec3 {
    transmittance(scene, *p + *n * RAY_OFFSET, *wi, 1e8, time, medium, rng)
}

/// Walks blockers along a shadow ray: opaque surfaces kill it, presence
/// cutouts attenuate, volume hulls toggle the medium, and each segment
/// picks up the medium's (possibly ratio-tracked) transmittance —
/// shadows through colored fog come out colored.
#[allow(clippy::too_many_arguments)]
fn transmittance(
    scene: &Scene,
    mut origin: Point3,
    dir: Vec3,
    mut remaining: f64,
    time: f64,
    medium: Option<u32>,
    rng: &mut Pcg32,
) -> Vec3 {
    // Shadow rays inherit the shading point's medium as the stack top;
    // hull crossings along the way push/pop from there.
    let mut med_stack = MediumStack::new(scene.atmosphere);
    if let Some(m) = medium {
        if Some(m) != scene.atmosphere {
            med_stack.push(m);
        }
    }
    let mut vis = Vec3::one();
    for _ in 0..MAX_PRESENCE_SKIPS {
        let ray = Ray::new(origin, dir).with_time(time);
        let hit = scene.intersect(&ray);
        let seg = hit.as_ref().map(|h| h.t).unwrap_or(remaining).min(remaining);
        if let Some(mid) = med_stack.current() {
            vis = vis
                * volume::transmittance(&scene.media[mid as usize], &origin, &dir, seg, rng);
            if max_component(&vis) < 1e-4 {
                return Vec3::zero();
            }
        }
        let Some(hit) = hit else {
            return vis;
        };
        if hit.t >= remaining {
            return vis;
        }
        let material = &scene.materials[hit.material_id];
        // Volume hull: pass through, pushing/popping the medium.
        if material.interior.is_some()
            && material.hair.is_none()
            && material.pbr.lobe_weights().is_none()
            && max_component(&(material.emission + material.pbr.glow)) <= 0.0
        {
            if hit.front_face {
                med_stack.push(material.interior.unwrap());
            } else {
                med_stack.pop();
            }
            origin = hit.point + dir * RAY_OFFSET;
            remaining -= hit.t + RAY_OFFSET;
            continue;
        }
        let presence = shade_params(scene, material, &hit, &dir, 0.0).presence.clamp(0.0, 1.0);
        if presence >= 1.0 {
            return Vec3::zero();
        }
        vis = vis * (1.0 - presence);
        if max_component(&vis) < 1e-4 {
            return Vec3::zero();
        }
        origin = hit.point + dir * RAY_OFFSET;
        remaining -= hit.t + RAY_OFFSET;
    }
    vis
}
