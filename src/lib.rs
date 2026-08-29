pub mod accel;
pub mod math;
pub mod parser;
pub mod geometry;
pub mod raytracer;
pub mod scene;
pub mod shading;
pub mod output;
pub mod texture;

pub use math::{Vec3, Point3, Matrix4};
pub use raytracer::Ray;
