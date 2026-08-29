//! GPU rendering backend for Apple Silicon using MLX (via mlx-rs).
//!
//! Wavefront architecture: all rays for a chunk of pixels live in big f32
//! arrays; intersections and shading are batched array ops that MLX executes
//! on the GPU. Reflections are an iterative masked bounce loop rather than
//! recursion. Computation is f32 (Metal has no f64), so output can differ
//! from the CPU backend by roughly a quantization step.

pub mod intersect;
mod renderer;
pub mod scene_arrays;
pub mod shade;

pub use renderer::render;

/// Minimum ray parameter, mirroring the CPU EPSILON hit threshold.
pub(crate) const T_EPS: f32 = 1e-6;
/// Shadow-ray origin offset along the normal (CPU uses 1e-4; f32 needs more
/// slack on large-coordinate scenes).
pub(crate) const SHADOW_EPS: f32 = 1e-4;
/// Reflection-ray origin offset along the normal.
pub(crate) const REFL_EPS: f32 = 1e-4;
