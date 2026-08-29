use super::Vec3;
use std::ops::{Add, Sub};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn origin() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    pub fn distance(&self, other: &Self) -> f64 {
        self.distance_squared(other).sqrt()
    }

    pub fn distance_squared(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        dx * dx + dy * dy + dz * dz
    }

    pub fn to_vec3(&self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
}

impl Add<Vec3> for Point3 {
    type Output = Self;

    fn add(self, vec: Vec3) -> Self {
        Self::new(self.x + vec.x, self.y + vec.y, self.z + vec.z)
    }
}

impl Sub<Vec3> for Point3 {
    type Output = Self;

    fn sub(self, vec: Vec3) -> Self {
        Self::new(self.x - vec.x, self.y - vec.y, self.z - vec.z)
    }
}

impl Sub<Point3> for Point3 {
    type Output = Vec3;

    fn sub(self, other: Point3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_point3_creation() {
        let p = Point3::new(1.0, 2.0, 3.0);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, 2.0);
        assert_eq!(p.z, 3.0);
    }

    #[test]
    fn test_point3_add_vec3() {
        let p = Point3::new(1.0, 2.0, 3.0);
        let v = Vec3::new(4.0, 5.0, 6.0);
        let result = p + v;
        assert_eq!(result, Point3::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn test_point3_sub_vec3() {
        let p = Point3::new(5.0, 7.0, 9.0);
        let v = Vec3::new(4.0, 5.0, 6.0);
        let result = p - v;
        assert_eq!(result, Point3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_point3_sub_point3() {
        let p1 = Point3::new(5.0, 7.0, 9.0);
        let p2 = Point3::new(1.0, 2.0, 3.0);
        let result = p1 - p2;
        assert_eq!(result, Vec3::new(4.0, 5.0, 6.0));
    }

    #[test]
    fn test_point3_distance() {
        let p1 = Point3::new(0.0, 0.0, 0.0);
        let p2 = Point3::new(3.0, 4.0, 0.0);
        assert_relative_eq!(p1.distance(&p2), 5.0);
    }
}
