use crate::math::Vec3;
use crate::raytracer::Intersection;
use crate::scene::{Material, MaterialType, Scene};

pub fn shade(intersection: &Intersection, scene: &Scene) -> Vec3 {
    let material = &scene.materials[intersection.material_id];

    let mut color = material.color * material.ka;

    // If no lights, use headlight shading (light from camera) for depth perception
    if scene.lights.is_empty() {
        let view_dir = (scene.camera.eye - intersection.point).normalize();
        let n_dot_v = intersection.normal.dot(&view_dir).max(0.0);
        color = color + material.color * (material.kd * n_dot_v);
        return color;
    }

    for light in &scene.lights {
        // Area lights are path-tracer only; Whitted shading skips them.
        if matches!(light.light_type, crate::scene::LightType::Rect { .. }) {
            continue;
        }
        let light_dir = light.direction_from(&intersection.point);
        let n_dot_l = intersection.normal.dot(&light_dir);
        if n_dot_l <= 0.0 {
            continue;
        }

        // Shadow ray: skip this light's contribution if occluded
        if scene.is_occluded(&intersection.point, &intersection.normal, light) {
            continue;
        }

        let diffuse = material.color * light.color * (material.kd * n_dot_l * light.intensity);
        color = color + diffuse;

        if material.ks > 0.0 {
            let view_dir = (scene.camera.eye - intersection.point).normalize();
            let half_vec = (light_dir + view_dir).normalize();
            let n_dot_h = intersection.normal.dot(&half_vec).max(0.0);
            let shininess = shininess_for(material);

            // Metals tint specular highlights; dielectrics reflect the light's color
            let specular_color = match material.material_type {
                MaterialType::Metal { .. } => material.color * light.color,
                _ => light.color,
            };
            let specular =
                specular_color * (material.ks * n_dot_h.powf(shininess) * light.intensity);
            color = color + specular;
        }
    }

    color
}

pub fn shininess_for(material: &Material) -> f64 {
    match material.material_type {
        MaterialType::Plastic { roughness } => 20.0 / roughness.max(0.1),
        MaterialType::Metal { roughness } => 50.0 / roughness.max(0.1),
        _ => 32.0,
    }
}
