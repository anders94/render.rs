use crate::geometry::{Intersectable, PrimitiveDesc, PrimitiveKind};
use crate::math::{Matrix4, Point3, Vec3, EPSILON};
use crate::raytracer::{Intersection, Ray};

pub struct Cone {
    pub height: f64,
    pub radius: f64,
    pub thetamax: f64,
    pub material_id: usize,
    pub transform: Matrix4,
    pub inverse_transform: Matrix4,
}

impl Cone {
    pub fn new(
        height: f64,
        radius: f64,
        thetamax: f64,
        material_id: usize,
        transform: Matrix4,
    ) -> Self {
        let inverse_transform = transform.inverse().unwrap_or(Matrix4::identity());
        Self {
            height,
            radius,
            thetamax,
            material_id,
            transform,
            inverse_transform,
        }
    }
}

impl Intersectable for Cone {
    fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        let local_origin = self.inverse_transform.transform_point(&ray.origin);
        let local_direction = self.inverse_transform.transform_vec(&ray.direction).normalize();
        let local_ray = Ray::new(local_origin, local_direction);

        let k = (self.radius / self.height) * (self.radius / self.height);

        let ox = local_ray.origin.x;
        let oy = local_ray.origin.y;
        let oz = local_ray.origin.z;
        let dx = local_ray.direction.x;
        let dy = local_ray.direction.y;
        let dz = local_ray.direction.z;

        let a = dx * dx + dy * dy - k * dz * dz;
        let b = 2.0 * (ox * dx + oy * dy - k * oz * dz);
        let c = ox * ox + oy * oy - k * oz * oz;

        if a.abs() < EPSILON {
            if b.abs() < EPSILON {
                return None;
            }
            let t = -c / b;
            if t > EPSILON {
                let local_hit_point = local_ray.at(t);
                if local_hit_point.z >= 0.0 && local_hit_point.z <= self.height {
                    return self.create_intersection(local_hit_point, ray);
                }
            }
            return None;
        }

        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            return None;
        }

        let sqrt_discriminant = discriminant.sqrt();
        let t1 = (-b - sqrt_discriminant) / (2.0 * a);
        let t2 = (-b + sqrt_discriminant) / (2.0 * a);

        for &t in &[t1, t2] {
            if t > EPSILON {
                let local_hit_point = local_ray.at(t);
                if local_hit_point.z >= 0.0 && local_hit_point.z <= self.height {
                    if self.thetamax < 360.0 {
                        let phi = local_hit_point.y.atan2(local_hit_point.x).to_degrees();
                        let phi = if phi < 0.0 { phi + 360.0 } else { phi };
                        if phi > self.thetamax {
                            continue;
                        }
                    }
                    return self.create_intersection(local_hit_point, ray);
                }
            }
        }

        None
    }

    fn describe(&self) -> PrimitiveDesc {
        PrimitiveDesc {
            kind: PrimitiveKind::Cone {
                height: self.height,
                radius: self.radius,
                thetamax: self.thetamax,
            },
            material_id: self.material_id,
            transform: self.transform,
            inverse_transform: self.inverse_transform,
        }
    }
}

impl Cone {
    fn create_intersection(
        &self,
        local_hit_point: Point3,
        world_ray: &Ray,
    ) -> Option<Intersection> {
        let world_hit_point = self.transform.transform_point(&local_hit_point);
        let t_world = world_hit_point.distance(&world_ray.origin);

        let r = (local_hit_point.x * local_hit_point.x + local_hit_point.y * local_hit_point.y).sqrt();
        let k = self.radius / self.height;
        let local_normal = Vec3::new(
            local_hit_point.x / r,
            local_hit_point.y / r,
            -k,
        )
        .normalize();

        let world_normal = self.inverse_transform.transform_normal(&local_normal).normalize();

        let mut phi = local_hit_point.y.atan2(local_hit_point.x).to_degrees();
        if phi < 0.0 {
            phi += 360.0;
        }
        let u = (phi / self.thetamax).clamp(0.0, 1.0);
        let v = (local_hit_point.z / self.height.max(1e-12)).clamp(0.0, 1.0);
        let dpdu = r.max(1e-6 * self.radius.max(1e-12)) * self.thetamax.to_radians();
        let slant = (self.height * self.height + self.radius * self.radius).sqrt();
        let density =
            1.0 / (dpdu * slant).max(1e-24).sqrt() / self.transform.approx_scale();

        Some(
            Intersection::new(t_world, world_hit_point, world_normal, self.material_id)
                .with_st([u, v], density),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cone_intersection() {
        let cone = Cone::new(2.0, 1.0, 360.0, 0, Matrix4::identity());
        let ray = Ray::new(Point3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));

        let intersection = cone.intersect(&ray);
        assert!(intersection.is_some());
    }

    #[test]
    fn test_cone_miss() {
        let cone = Cone::new(2.0, 1.0, 360.0, 0, Matrix4::identity());
        let ray = Ray::new(Point3::new(5.0, 5.0, -5.0), Vec3::new(0.0, 0.0, 1.0));

        let intersection = cone.intersect(&ray);
        assert!(intersection.is_none());
    }
}
