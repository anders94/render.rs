use crate::geometry::{Intersectable, PrimitiveDesc, PrimitiveKind};
use crate::math::{Matrix4, Point3, EPSILON};
use crate::raytracer::{Intersection, Ray};

pub struct Sphere {
    pub radius: f64,
    pub zmin: f64,
    pub zmax: f64,
    pub thetamax: f64,
    pub material_id: usize,
    pub transform: Matrix4,
    pub inverse_transform: Matrix4,
}

impl Sphere {
    pub fn new(
        radius: f64,
        zmin: f64,
        zmax: f64,
        thetamax: f64,
        material_id: usize,
        transform: Matrix4,
    ) -> Self {
        let inverse_transform = transform.inverse().unwrap_or(Matrix4::identity());
        Self {
            radius,
            zmin,
            zmax,
            thetamax,
            material_id,
            transform,
            inverse_transform,
        }
    }
}

impl Intersectable for Sphere {
    fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        let local_origin = self.inverse_transform.transform_point(&ray.origin);
        let local_direction = self.inverse_transform.transform_vec(&ray.direction).normalize();
        let local_ray = Ray::new(local_origin, local_direction);

        let oc = local_ray.origin - Point3::origin();
        let a = local_ray.direction.dot(&local_ray.direction);
        let b = 2.0 * oc.dot(&local_ray.direction);
        let c = oc.dot(&oc) - self.radius * self.radius;

        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            return None;
        }

        let sqrt_discriminant = discriminant.sqrt();
        let t1 = (-b - sqrt_discriminant) / (2.0 * a);
        let t2 = (-b + sqrt_discriminant) / (2.0 * a);

        // Near root, falling back to the far root when the ray starts
        // inside the sphere (required for glass refraction).
        let t = if t1 > EPSILON {
            t1
        } else if t2 > EPSILON {
            t2
        } else {
            return None;
        };

        let local_hit_point = local_ray.at(t);

        if local_hit_point.z < self.zmin || local_hit_point.z > self.zmax {
            return None;
        }

        if self.thetamax < 360.0 {
            let phi = local_hit_point.y.atan2(local_hit_point.x).to_degrees();
            let phi = if phi < 0.0 { phi + 360.0 } else { phi };
            if phi > self.thetamax {
                return None;
            }
        }

        let world_hit_point = self.transform.transform_point(&local_hit_point);
        let t_world = world_hit_point.distance(&ray.origin);

        let local_normal = (local_hit_point - Point3::origin()).normalize();
        let world_normal = self.inverse_transform.transform_normal(&local_normal).normalize();

        Some(
            Intersection::new(t_world, world_hit_point, world_normal, self.material_id)
                .with_front_face(ray.direction.dot(&world_normal) < 0.0),
        )
    }

    fn describe(&self) -> PrimitiveDesc {
        PrimitiveDesc {
            kind: PrimitiveKind::Sphere {
                radius: self.radius,
                zmin: self.zmin,
                zmax: self.zmax,
                thetamax: self.thetamax,
            },
            material_id: self.material_id,
            transform: self.transform,
            inverse_transform: self.inverse_transform,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;

    #[test]
    fn test_sphere_intersection() {
        let sphere = Sphere::new(1.0, -1.0, 1.0, 360.0, 0, Matrix4::identity());
        let ray = Ray::new(Point3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));

        let intersection = sphere.intersect(&ray);
        assert!(intersection.is_some());

        let intersection = intersection.unwrap();
        assert!(intersection.t > 0.0);
    }

    #[test]
    fn test_sphere_miss() {
        let sphere = Sphere::new(1.0, -1.0, 1.0, 360.0, 0, Matrix4::identity());
        let ray = Ray::new(Point3::new(5.0, 5.0, -5.0), Vec3::new(0.0, 0.0, 1.0));

        let intersection = sphere.intersect(&ray);
        assert!(intersection.is_none());
    }
}
