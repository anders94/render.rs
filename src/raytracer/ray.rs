use crate::math::{Point3, Vec3};

#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Point3,
    pub direction: Vec3,
    /// Shutter time in [0,1); motion-blurred primitives interpolate their
    /// endpoints by this. 0 for rays that don't carry a time.
    pub time: f64,
}

impl Ray {
    pub fn new(origin: Point3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
            time: 0.0,
        }
    }

    pub fn with_time(mut self, time: f64) -> Self {
        self.time = time;
        self
    }

    pub fn at(&self, t: f64) -> Point3 {
        self.origin + self.direction * t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ray_creation() {
        let origin = Point3::new(0.0, 0.0, 0.0);
        let direction = Vec3::new(1.0, 0.0, 0.0);
        let ray = Ray::new(origin, direction);

        assert_eq!(ray.origin, origin);
        assert_relative_eq!(ray.direction.length(), 1.0);
    }

    #[test]
    fn test_ray_at() {
        let ray = Ray::new(
            Point3::new(1.0, 2.0, 3.0),
            Vec3::new(1.0, 0.0, 0.0),
        );

        let p = ray.at(5.0);
        assert_eq!(p, Point3::new(6.0, 2.0, 3.0));
    }
}
