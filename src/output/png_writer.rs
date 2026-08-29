use crate::output::{clamp, gamma_correct, Image};
use anyhow::Result;
use image::{ImageBuffer, Rgb};

pub fn write_png(image: &Image, filename: &str) -> Result<()> {
    let height = image.len() as u32;
    let width = if height > 0 { image[0].len() as u32 } else { 0 };

    let mut img_buffer = ImageBuffer::new(width, height);

    for (y, row) in image.iter().enumerate() {
        for (x, pixel) in row.iter().enumerate() {
            let corrected = gamma_correct(*pixel, 2.2);

            let r = (clamp(corrected.x, 0.0, 1.0) * 255.0) as u8;
            let g = (clamp(corrected.y, 0.0, 1.0) * 255.0) as u8;
            let b = (clamp(corrected.z, 0.0, 1.0) * 255.0) as u8;

            img_buffer.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }

    img_buffer.save(filename)?;

    Ok(())
}
