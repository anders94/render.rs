use crate::output::{clamp, gamma_correct, Image};
use anyhow::Result;
use std::fs::File;
use std::io::{BufWriter, Write};

/// Write a binary PPM (P6) file — the default, much smaller than ASCII.
pub fn write_ppm(image: &Image, filename: &str) -> Result<()> {
    let mut file = BufWriter::new(File::create(filename)?);

    let height = image.len();
    let width = if height > 0 { image[0].len() } else { 0 };

    write!(file, "P6\n{} {}\n255\n", width, height)?;

    let mut buf = Vec::with_capacity(width * height * 3);
    for row in image {
        for pixel in row {
            let [r, g, b] = pixel_to_rgb(pixel);
            buf.extend_from_slice(&[r, g, b]);
        }
    }
    file.write_all(&buf)?;

    Ok(())
}

/// Write an ASCII PPM (P3) file.
pub fn write_ppm_ascii(image: &Image, filename: &str) -> Result<()> {
    let mut file = BufWriter::new(File::create(filename)?);

    let height = image.len();
    let width = if height > 0 { image[0].len() } else { 0 };

    writeln!(file, "P3")?;
    writeln!(file, "{} {}", width, height)?;
    writeln!(file, "255")?;

    for row in image {
        for pixel in row {
            let [r, g, b] = pixel_to_rgb(pixel);
            writeln!(file, "{} {} {}", r, g, b)?;
        }
    }

    Ok(())
}

fn pixel_to_rgb(pixel: &crate::math::Vec3) -> [u8; 3] {
    let corrected = gamma_correct(*pixel, 2.2);
    [
        (clamp(corrected.x, 0.0, 1.0) * 255.0) as u8,
        (clamp(corrected.y, 0.0, 1.0) * 255.0) as u8,
        (clamp(corrected.z, 0.0, 1.0) * 255.0) as u8,
    ]
}
