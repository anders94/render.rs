//! Linear OpenEXR output (via the `image` crate). No gamma, no
//! quantization: the film's linear radiance goes to disk as f32, which is
//! what compositors expect.

use crate::output::Image;
use anyhow::Result;

pub fn write_exr(image: &Image, filename: &str) -> Result<()> {
    let height = image.len() as u32;
    let width = if height > 0 { image[0].len() as u32 } else { 0 };

    let mut buffer = image::Rgb32FImage::new(width, height);
    for (y, row) in image.iter().enumerate() {
        for (x, pixel) in row.iter().enumerate() {
            buffer.put_pixel(
                x as u32,
                y as u32,
                image::Rgb([pixel.x as f32, pixel.y as f32, pixel.z as f32]),
            );
        }
    }
    image::DynamicImage::ImageRgb32F(buffer).save_with_format(filename, image::ImageFormat::OpenExr)?;
    Ok(())
}
