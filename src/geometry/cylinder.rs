use crate::geometry::{Intersectable, PrimitiveDesc, PrimitiveKind};
use crate::math::{Matrix4, Vec3, EPSILON};
use crate::raytracer::{Intersection, Ray};

pub struct Cylinder {
    pub radius: f64,
    pub zmin: f64,
    pub zmax: f64,
    pub thetamax: f64,
    pub material_id: usize,
    pub transform: Matrix4,
    pub inverse_transform: Matrix4,
}

impl Cylinder {
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

impl Intersectable for Cylinder {
    fn intersect(&self, ray: &Ray) -> Option<Intersection> {
        let local_origin = self.inverse_transform.transform_point(&ray.origin);
        let local_direction = self.inverse_transform.transform_vec(&ray.direction).normalize();
        let local_ray = Ray::new(local_origin, local_direction);

        let ox = local_ray.origin.x;
        let oy = local_ray.origin.y;
        let dx = local_ray.direction.x;
        let dy = local_ray.direction.y;

        let a = dx * dx + dy * dy;
        let b = 2.0 * (ox * dx + oy * dy);
        let c = ox * ox + oy * oy - self.radius * self.radius;

        if a.abs() < EPSILON {
            return None;
        }

        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            return None;
        }

        let sqrt_discriminant = discriminant.sqrt();
        let t1 = (-b - sqrt_discriminant) / (2.0 * a);
        let t2 = (-b + sqrt_discriminant) / (2.0 * a);

        let mut t = if t1 > EPSILON { t1 } else { t2 };
        if t <= EPSILON {
            return None;
        }

        let local_hit_point = local_ray.at(t);

        if local_hit_point.z < self.zmin || local_hit_point.z > self.zmax {
            if t == t1 && t2 > EPSILON {
                t = t2;
                let local_hit_point2 = local_ray.at(t);
                if local_hit_point2.z < self.zmin || local_hit_point2.z > self.zmax {
                    return None;
                }
            } else {
                return None;
            }
        }

        let local_hit_point = local_ray.at(t);

        if self.thetamax < 360.0 {
            let phi = local_hit_point.y.atan2(local_hit_point.x).to_degrees();
            let phi = if phi < 0.0 { phi + 360.0 } else { phi };
            if phi > self.thetamax {
                return None;
            }
        }

        let world_hit_point = self.transform.transform_point(&local_hit_point);
        let t_world = world_hit_point.distance(&ray.origin);

        let local_normal = Vec3::new(local_hit_point.x, local_hit_point.y, 0.0).normalize();
        let world_normal = self.inverse_transform.transform_normal(&local_normal).normalize();

        Some(Intersection::new(
            t_world,
            world_hit_point,
            world_normal,
            self.material_id,
        ))
    }

    fn describe(&self) -> PrimitiveDesc {
        PrimitiveDesc {
            kind: PrimitiveKind::Cylinder {
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
    use crate::math::Point3;

    #[test]
    fn test_cylinder_intersection() {
        let cylinder = Cylinder::new(1.0, -1.0, 1.0, 360.0, 0, Matrix4::identity());
        let ray = Ray::new(Point3::new(2.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0));

        let intersection = cylinder.intersect(&ray);
        assert!(intersection.is_some());
    }

    #[test]
    fn test_cylinder_miss() {
        let cylinder = Cylinder::new(1.0, -1.0, 1.0, 360.0, 0, Matrix4::identity());
        let ray = Ray::new(Point3::new(5.0, 5.0, -5.0), Vec3::new(0.0, 0.0, 1.0));

        let intersection = cylinder.intersect(&ray);
        assert!(intersection.is_none());
    }
}
