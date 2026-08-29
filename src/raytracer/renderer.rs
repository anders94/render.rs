use crate::math::Vec3;
use crate::output::Image;
use crate::raytracer::Ray;
use crate::scene::{MaterialType, Scene};
use crate::shading;
use rayon::prelude::*;

const MAX_DEPTH: u32 = 5;

pub fn render(scene: &Scene) -> Image {
    let width = scene.camera.width;
    let height = scene.camera.height;
    let (samples_x, samples_y) = scene.pixel_samples;
    let sample_count = (samples_x * samples_y) as f64;

    let pixels: Vec<Vec<Vec3>> = (0..height)
        .into_par_iter()
        .map(|y| {
            (0..width)
                .map(|x| {
                    // Stratified supersampling: one ray through the center of
                    // each subpixel cell, averaged.
                    let mut color = Vec3::zero();
                    for sy in 0..samples_y {
                        for sx in 0..samples_x {
                            let px = x as f64 + (sx as f64 + 0.5) / samples_x as f64;
                            let py = y as f64 + (sy as f64 + 0.5) / samples_y as f64;
                            let ray = scene.camera.generate_ray(px, py);
                            color = color + trace_ray(&ray, scene, 0);
                        }
                    }
                    color / sample_count
                })
                .collect()
        })
        .collect();

    pixels
}

fn trace_ray(ray: &Ray, scene: &Scene, depth: u32) -> Vec3 {
    if depth >= MAX_DEPTH {
        return scene.background_color;
    }

    if let Some(intersection) = scene.intersect(ray) {
        let color = shading::shade(&intersection, scene);

        let material = &scene.materials[intersection.material_id];
        let reflectivity = material.reflectivity();
        if reflectivity > 0.0 {
            let reflection_dir = ray.direction.reflect(&intersection.normal);
            let reflection_origin = intersection.point + intersection.normal * 1e-4;
            let reflection_ray = Ray::new(reflection_origin, reflection_dir);

            let reflection_color = trace_ray(&reflection_ray, scene, depth + 1);
            // Metals tint their reflections; dielectrics like plastic don't.
            let tint = match material.material_type {
                MaterialType::Metal { .. } => material.color,
                _ => Vec3::one(),
            };
            color * (1.0 - reflectivity) + reflection_color * tint * reflectivity
        } else {
            color
        }
    } else {
        scene.background_color
    }
}
