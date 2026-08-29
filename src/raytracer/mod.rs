mod ray;
mod intersection;
pub mod flatten;
pub mod renderer;
#[cfg(feature = "mlx")]
pub mod mlx;
#[cfg(target_os = "macos")]
pub mod metal;

pub use ray::Ray;
pub use intersection::Intersection;
