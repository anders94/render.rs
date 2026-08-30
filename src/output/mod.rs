mod exr_writer;
pub mod film;
pub mod denoise;
pub mod multilayer;
pub mod oidn;
pub mod tonemap;
mod ppm;
mod png_writer;

pub use exr_writer::write_exr;
pub use film::{AuxSample, Film};
pub use denoise::denoise;
pub use multilayer::write_multilayer_exr;
pub use tonemap::{apply_image as apply_tonemap, Tonemap};
pub use ppm::{write_ppm, write_ppm_ascii};
pub use png_writer::write_png;

use crate::math::Vec3;

pub type Image = Vec<Vec<Vec3>>;

pub fn clamp(value: f64, min: f64, max: f64) -> f64 {
    value.max(min).min(max)
}

pub fn gamma_correct(color: Vec3, gamma: f64) -> Vec3 {
    Vec3::new(
        color.x.powf(1.0 / gamma),
        color.y.powf(1.0 / gamma),
        color.z.powf(1.0 / gamma),
    )
}
