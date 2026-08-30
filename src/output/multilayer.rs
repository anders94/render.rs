//! Multilayer OpenEXR output (roadmap Phase 11): the full Film — beauty,
//! diffuse/specular split, albedo, normal, depth, and object id — as one
//! EXR with named layers, the way compositors expect production renders.
//! The id manifest rides along as a header string attribute
//! ("render_rs/manifest": "id:name;..."), cryptomatte-in-spirit.

use crate::output::film::Film;
use anyhow::Result;
use exr::prelude::*;

pub fn write_multilayer_exr(film: &Film, filename: &str) -> Result<()> {
    let w = film.width();
    let h = film.height();

    // exr's AnyChannels wants per-channel flat sample vectors.
    let plane = |img: &crate::output::Image, c: usize| -> FlatSamples {
        let mut v = Vec::with_capacity(w * h);
        for row in img {
            for p in row {
                v.push(match c {
                    0 => p.x as f32,
                    1 => p.y as f32,
                    _ => p.z as f32,
                });
            }
        }
        FlatSamples::F32(v)
    };

    let chan = |name: &str, samples: FlatSamples| AnyChannel::new(name, samples);

    // One layer, compositor-style dotted channel names (Nuke/Natron split
    // on the prefix): R/G/B beauty + <layer>.<chan> for the rest.
    let mut channels = vec![
        chan("R", plane(&film.beauty, 0)),
        chan("G", plane(&film.beauty, 1)),
        chan("B", plane(&film.beauty, 2)),
        chan("diffuse.R", plane(&film.diffuse, 0)),
        chan("diffuse.G", plane(&film.diffuse, 1)),
        chan("diffuse.B", plane(&film.diffuse, 2)),
        chan("specular.R", plane(&film.specular, 0)),
        chan("specular.G", plane(&film.specular, 1)),
        chan("specular.B", plane(&film.specular, 2)),
        chan("albedo.R", plane(&film.albedo, 0)),
        chan("albedo.G", plane(&film.albedo, 1)),
        chan("albedo.B", plane(&film.albedo, 2)),
        chan("N.X", plane(&film.normal, 0)),
        chan("N.Y", plane(&film.normal, 1)),
        chan("N.Z", plane(&film.normal, 2)),
        chan("depth.Z", plane(&film.depth, 0)),
        chan("id.x", plane(&film.id, 0)),
        chan("id.coverage", plane(&film.id, 1)),
    ];
    // exr requires channels sorted by name.
    channels.sort_by(|a, b| a.name.cmp(&b.name));

    let manifest: String = film
        .manifest
        .iter()
        .map(|(id, name)| format!("{id}:{name}"))
        .collect::<Vec<_>>()
        .join(";");

    let mut attributes = LayerAttributes::named("rgba");
    attributes.other.insert(
        Text::from("render_rs/manifest"),
        AttributeValue::Text(Text::new_or_panic(manifest.as_str())),
    );

    let layer = Layer::new(
        (w, h),
        attributes,
        Encoding::FAST_LOSSLESS,
        AnyChannels::sort(SmallVec::from_vec(channels)),
    );

    Image::from_layer(layer)
        .write()
        .to_file(filename)
        .map_err(|e| anyhow::anyhow!("EXR write failed: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;
    use std::collections::BTreeMap;

    #[test]
    fn multilayer_round_trip() {
        let (w, h) = (8, 6);
        let grad = |f: f64| {
            (0..h)
                .map(|y| {
                    (0..w)
                        .map(|x| Vec3::new(x as f64 * f, y as f64 * f, 0.25))
                        .collect()
                })
                .collect::<Vec<Vec<Vec3>>>()
        };
        let mut manifest = BTreeMap::new();
        manifest.insert(1, "hero".to_string());
        manifest.insert(2, "floor".to_string());
        let film = Film {
            beauty: grad(0.1),
            diffuse: grad(0.05),
            specular: grad(0.05),
            albedo: grad(0.08),
            normal: grad(0.02),
            depth: grad(1.0),
            id: grad(1.0),
            manifest,
        };
        let path = std::env::temp_dir().join("render_rs_multilayer.exr");
        write_multilayer_exr(&film, path.to_str().unwrap()).unwrap();

        // Read back and verify channels + a pixel value.
        let img = exr::prelude::read()
            .no_deep_data()
            .largest_resolution_level()
            .all_channels()
            .all_layers()
            .all_attributes()
            .from_file(&path)
            .unwrap();
        let layer = &img.layer_data[0];
        let names: Vec<String> = layer
            .channel_data
            .list
            .iter()
            .map(|c| c.name.to_string())
            .collect();
        for expect in ["R", "diffuse.R", "specular.B", "albedo.G", "N.Z", "depth.Z", "id.x"] {
            assert!(names.iter().any(|n| n == expect), "missing channel {expect}: {names:?}");
        }
        // Beauty R at (3, 2) = 0.3.
        let r = layer
            .channel_data
            .list
            .iter()
            .find(|c| c.name.to_string() == "R")
            .unwrap();
        if let FlatSamples::F32(v) = &r.sample_data {
            let val = v[2 * 8 + 3];
            assert!((val - 0.3).abs() < 1e-6, "beauty R = {val}");
        } else {
            panic!("expected f32 samples");
        }
        std::fs::remove_file(&path).ok();
    }
}
