use super::{Point3, Vec3};
use std::ops::Mul;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix4 {
    m: [[f64; 4]; 4],
}

impl Matrix4 {
    pub fn new(m: [[f64; 4]; 4]) -> Self {
        Self { m }
    }

    pub fn rows(&self) -> [[f64; 4]; 4] {
        self.m
    }

    pub fn identity() -> Self {
        Self::new([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    pub fn translate(x: f64, y: f64, z: f64) -> Self {
        Self::new([
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    pub fn scale(x: f64, y: f64, z: f64) -> Self {
        Self::new([
            [x, 0.0, 0.0, 0.0],
            [0.0, y, 0.0, 0.0],
            [0.0, 0.0, z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    pub fn rotate_x(angle: f64) -> Self {
        let rad = angle.to_radians();
        let cos = rad.cos();
        let sin = rad.sin();
        Self::new([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos, -sin, 0.0],
            [0.0, sin, cos, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    pub fn rotate_y(angle: f64) -> Self {
        let rad = angle.to_radians();
        let cos = rad.cos();
        let sin = rad.sin();
        Self::new([
            [cos, 0.0, sin, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sin, 0.0, cos, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    pub fn rotate_z(angle: f64) -> Self {
        let rad = angle.to_radians();
        let cos = rad.cos();
        let sin = rad.sin();
        Self::new([
            [cos, -sin, 0.0, 0.0],
            [sin, cos, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    pub fn rotate(angle: f64, x: f64, y: f64, z: f64) -> Self {
        let axis = Vec3::new(x, y, z).normalize();
        let rad = angle.to_radians();
        let cos = rad.cos();
        let sin = rad.sin();
        let one_minus_cos = 1.0 - cos;

        let x = axis.x;
        let y = axis.y;
        let z = axis.z;

        Self::new([
            [
                cos + x * x * one_minus_cos,
                x * y * one_minus_cos - z * sin,
                x * z * one_minus_cos + y * sin,
                0.0,
            ],
            [
                y * x * one_minus_cos + z * sin,
                cos + y * y * one_minus_cos,
                y * z * one_minus_cos - x * sin,
                0.0,
            ],
            [
                z * x * one_minus_cos - y * sin,
                z * y * one_minus_cos + x * sin,
                cos + z * z * one_minus_cos,
                0.0,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    pub fn inverse(&self) -> Option<Self> {
        let mut inv = [[0.0; 4]; 4];
        let m = &self.m;

        inv[0][0] = m[1][1] * m[2][2] * m[3][3] - m[1][1] * m[2][3] * m[3][2] -
                    m[2][1] * m[1][2] * m[3][3] + m[2][1] * m[1][3] * m[3][2] +
                    m[3][1] * m[1][2] * m[2][3] - m[3][1] * m[1][3] * m[2][2];

        inv[1][0] = -m[1][0] * m[2][2] * m[3][3] + m[1][0] * m[2][3] * m[3][2] +
                     m[2][0] * m[1][2] * m[3][3] - m[2][0] * m[1][3] * m[3][2] -
                     m[3][0] * m[1][2] * m[2][3] + m[3][0] * m[1][3] * m[2][2];

        inv[2][0] = m[1][0] * m[2][1] * m[3][3] - m[1][0] * m[2][3] * m[3][1] -
                    m[2][0] * m[1][1] * m[3][3] + m[2][0] * m[1][3] * m[3][1] +
                    m[3][0] * m[1][1] * m[2][3] - m[3][0] * m[1][3] * m[2][1];

        inv[3][0] = -m[1][0] * m[2][1] * m[3][2] + m[1][0] * m[2][2] * m[3][1] +
                     m[2][0] * m[1][1] * m[3][2] - m[2][0] * m[1][2] * m[3][1] -
                     m[3][0] * m[1][1] * m[2][2] + m[3][0] * m[1][2] * m[2][1];

        let det = m[0][0] * inv[0][0] + m[0][1] * inv[1][0] + m[0][2] * inv[2][0] + m[0][3] * inv[3][0];

        if det.abs() < super::EPSILON {
            return None;
        }

        inv[0][1] = -m[0][1] * m[2][2] * m[3][3] + m[0][1] * m[2][3] * m[3][2] +
                     m[2][1] * m[0][2] * m[3][3] - m[2][1] * m[0][3] * m[3][2] -
                     m[3][1] * m[0][2] * m[2][3] + m[3][1] * m[0][3] * m[2][2];

        inv[1][1] = m[0][0] * m[2][2] * m[3][3] - m[0][0] * m[2][3] * m[3][2] -
                    m[2][0] * m[0][2] * m[3][3] + m[2][0] * m[0][3] * m[3][2] +
                    m[3][0] * m[0][2] * m[2][3] - m[3][0] * m[0][3] * m[2][2];

        inv[2][1] = -m[0][0] * m[2][1] * m[3][3] + m[0][0] * m[2][3] * m[3][1] +
                     m[2][0] * m[0][1] * m[3][3] - m[2][0] * m[0][3] * m[3][1] -
                     m[3][0] * m[0][1] * m[2][3] + m[3][0] * m[0][3] * m[2][1];

        inv[3][1] = m[0][0] * m[2][1] * m[3][2] - m[0][0] * m[2][2] * m[3][1] -
                    m[2][0] * m[0][1] * m[3][2] + m[2][0] * m[0][2] * m[3][1] +
                    m[3][0] * m[0][1] * m[2][2] - m[3][0] * m[0][2] * m[2][1];

        inv[0][2] = m[0][1] * m[1][2] * m[3][3] - m[0][1] * m[1][3] * m[3][2] -
                    m[1][1] * m[0][2] * m[3][3] + m[1][1] * m[0][3] * m[3][2] +
                    m[3][1] * m[0][2] * m[1][3] - m[3][1] * m[0][3] * m[1][2];

        inv[1][2] = -m[0][0] * m[1][2] * m[3][3] + m[0][0] * m[1][3] * m[3][2] +
                     m[1][0] * m[0][2] * m[3][3] - m[1][0] * m[0][3] * m[3][2] -
                     m[3][0] * m[0][2] * m[1][3] + m[3][0] * m[0][3] * m[1][2];

        inv[2][2] = m[0][0] * m[1][1] * m[3][3] - m[0][0] * m[1][3] * m[3][1] -
                    m[1][0] * m[0][1] * m[3][3] + m[1][0] * m[0][3] * m[3][1] +
                    m[3][0] * m[0][1] * m[1][3] - m[3][0] * m[0][3] * m[1][1];

        inv[3][2] = -m[0][0] * m[1][1] * m[3][2] + m[0][0] * m[1][2] * m[3][1] +
                     m[1][0] * m[0][1] * m[3][2] - m[1][0] * m[0][2] * m[3][1] -
                     m[3][0] * m[0][1] * m[1][2] + m[3][0] * m[0][2] * m[1][1];

        inv[0][3] = -m[0][1] * m[1][2] * m[2][3] + m[0][1] * m[1][3] * m[2][2] +
                     m[1][1] * m[0][2] * m[2][3] - m[1][1] * m[0][3] * m[2][2] -
                     m[2][1] * m[0][2] * m[1][3] + m[2][1] * m[0][3] * m[1][2];

        inv[1][3] = m[0][0] * m[1][2] * m[2][3] - m[0][0] * m[1][3] * m[2][2] -
                    m[1][0] * m[0][2] * m[2][3] + m[1][0] * m[0][3] * m[2][2] +
                    m[2][0] * m[0][2] * m[1][3] - m[2][0] * m[0][3] * m[1][2];

        inv[2][3] = -m[0][0] * m[1][1] * m[2][3] + m[0][0] * m[1][3] * m[2][1] +
                     m[1][0] * m[0][1] * m[2][3] - m[1][0] * m[0][3] * m[2][1] -
                     m[2][0] * m[0][1] * m[1][3] + m[2][0] * m[0][3] * m[1][1];

        inv[3][3] = m[0][0] * m[1][1] * m[2][2] - m[0][0] * m[1][2] * m[2][1] -
                    m[1][0] * m[0][1] * m[2][2] + m[1][0] * m[0][2] * m[2][1] +
                    m[2][0] * m[0][1] * m[1][2] - m[2][0] * m[0][2] * m[1][1];

        let det_inv = 1.0 / det;
        for i in 0..4 {
            for j in 0..4 {
                inv[i][j] *= det_inv;
            }
        }

        Some(Self::new(inv))
    }

    pub fn transform_point(&self, p: &Point3) -> Point3 {
        let x = self.m[0][0] * p.x + self.m[0][1] * p.y + self.m[0][2] * p.z + self.m[0][3];
        let y = self.m[1][0] * p.x + self.m[1][1] * p.y + self.m[1][2] * p.z + self.m[1][3];
        let z = self.m[2][0] * p.x + self.m[2][1] * p.y + self.m[2][2] * p.z + self.m[2][3];
        let w = self.m[3][0] * p.x + self.m[3][1] * p.y + self.m[3][2] * p.z + self.m[3][3];

        if (w - 1.0).abs() < super::EPSILON {
            Point3::new(x, y, z)
        } else {
            Point3::new(x / w, y / w, z / w)
        }
    }

    pub fn transform_vec(&self, v: &Vec3) -> Vec3 {
        let x = self.m[0][0] * v.x + self.m[0][1] * v.y + self.m[0][2] * v.z;
        let y = self.m[1][0] * v.x + self.m[1][1] * v.y + self.m[1][2] * v.z;
        let z = self.m[2][0] * v.x + self.m[2][1] * v.y + self.m[2][2] * v.z;
        Vec3::new(x, y, z)
    }

    /// Transform a surface normal: multiplies by the transpose of this matrix's
    /// upper 3x3. Pass the *inverse* object transform to get correct normals
    /// under non-uniform scale.
    pub fn transform_normal(&self, n: &Vec3) -> Vec3 {
        let x = self.m[0][0] * n.x + self.m[1][0] * n.y + self.m[2][0] * n.z;
        let y = self.m[0][1] * n.x + self.m[1][1] * n.y + self.m[2][1] * n.z;
        let z = self.m[0][2] * n.x + self.m[1][2] * n.y + self.m[2][2] * n.z;
        Vec3::new(x, y, z)
    }

    /// Average length scale of the linear part (geometric mean of the
    /// transformed basis lengths) — the isotropic approximation used for
    /// texture-footprint estimates.
    pub fn approx_scale(&self) -> f64 {
        let lx = self.transform_vec(&Vec3::new(1.0, 0.0, 0.0)).length();
        let ly = self.transform_vec(&Vec3::new(0.0, 1.0, 0.0)).length();
        let lz = self.transform_vec(&Vec3::new(0.0, 0.0, 1.0)).length();
        (lx * ly * lz).cbrt().max(1e-12)
    }
}

impl Mul<Matrix4> for Matrix4 {
    type Output = Matrix4;

    fn mul(self, other: Matrix4) -> Matrix4 {
        let mut result = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                result[i][j] = 0.0;
                for k in 0..4 {
                    result[i][j] += self.m[i][k] * other.m[k][j];
                }
            }
        }
        Matrix4::new(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_identity() {
        let m = Matrix4::identity();
        let p = Point3::new(1.0, 2.0, 3.0);
        let result = m.transform_point(&p);
        assert_eq!(result, p);
    }

    #[test]
    fn test_translate() {
        let m = Matrix4::translate(1.0, 2.0, 3.0);
        let p = Point3::new(4.0, 5.0, 6.0);
        let result = m.transform_point(&p);
        assert_eq!(result, Point3::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn test_scale() {
        let m = Matrix4::scale(2.0, 3.0, 4.0);
        let p = Point3::new(1.0, 2.0, 3.0);
        let result = m.transform_point(&p);
        assert_eq!(result, Point3::new(2.0, 6.0, 12.0));
    }

    #[test]
    fn test_rotate_z() {
        let m = Matrix4::rotate_z(90.0);
        let p = Point3::new(1.0, 0.0, 0.0);
        let result = m.transform_point(&p);
        assert_relative_eq!(result.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.y, 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_multiply() {
        let t = Matrix4::translate(1.0, 0.0, 0.0);
        let s = Matrix4::scale(2.0, 2.0, 2.0);
        let combined = t * s;
        let p = Point3::new(1.0, 1.0, 1.0);
        let result = combined.transform_point(&p);
        assert_eq!(result, Point3::new(3.0, 2.0, 2.0));
    }
}
