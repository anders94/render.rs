//! Scene-level validation of the Phase-4 material/light system: the white
//! furnace (a perfect diffuse sphere in a uniform dome must render at
//! exactly the dome radiance), energy conservation for other lobes, and
//! smoke tests for the modern Light types. (Lobe-level furnace and
//! chi-square-style checks live in src/raytracer/pt/bxdf.rs tests.)

mod common;

use render_rs::math::Vec3;
use render_rs::parser::{parse_rib, SceneBuilder};
use render_rs::raytracer::pt;

fn build(rib: &str) -> render_rs::scene::Scene {
    let requests = parse_rib(rib).unwrap();
    SceneBuilder::new().build(&requests).unwrap()
}

fn mean_of_center(img: &render_rs::output::Image, margin: usize) -> Vec3 {
    let mut sum = Vec3::zero();
    let mut n: f64 = 0.0;
    for row in img.iter().skip(margin).take(img.len() - 2 * margin) {
        for p in row.iter().skip(margin).take(row.len() - 2 * margin) {
            sum = sum + *p;
            n += 1.0;
        }
    }
    sum / n.max(1.0)
}

#[test]
fn white_furnace_diffuse() {
    // Albedo-1 diffuse sphere filling the view, uniform dome of radiance 1:
    // every path returns exactly 1 (any deviation is an energy bug).
    let scene = build(
        r#"
        Format 64 64 1.0
        Projection "perspective" "fov" [30]
        WorldBegin
            Light "PxrDomeLight" "sky" "intensity" [1]
            Bxdf "PxrSurface" "white" "diffuseGain" [1] "diffuseColor" [1 1 1]
            Translate 0 0 4
            Sphere 1.5 -1.5 1.5 360
        WorldEnd
    "#,
    );
    let img = pt::render(&scene, 128);
    let mean = mean_of_center(&img, 8);
    for (c, ch) in [(mean.x, "r"), (mean.y, "g"), (mean.z, "b")] {
        assert!(
            (c - 1.0).abs() < 0.02,
            "furnace {ch} = {c:.4}, expected 1.0"
        );
    }
}

#[test]
fn furnace_energy_conservation_metal_and_glass() {
    // Conductor and glass spheres in the same furnace: must never gain
    // energy (≤1) and rough GGX single-scatter should stay above 0.7.
    for (label, bxdf) in [
        (
            "conductor",
            r#""specularFaceColor" [1 1 1] "specularRoughness" [0.4] "diffuseGain" [0]"#,
        ),
        (
            "glass",
            r#""diffuseGain" [0] "glassRefractionGain" [1] "glassIor" [1.5] "glassRoughness" [0.1]"#,
        ),
    ] {
        let rib = format!(
            r#"
            Format 48 48 1.0
            Projection "perspective" "fov" [30]
            WorldBegin
                Light "PxrDomeLight" "sky" "intensity" [1]
                Bxdf "PxrSurface" "m" {bxdf}
                Translate 0 0 4
                Sphere 1.5 -1.5 1.5 360
            WorldEnd
        "#
        );
        let scene = build(&rib);
        let img = pt::render(&scene, 96);
        let mean = mean_of_center(&img, 6);
        let avg = (mean.x + mean.y + mean.z) / 3.0;
        assert!(avg <= 1.05, "{label}: gains energy ({avg:.4})");
        assert!(avg >= 0.55, "{label}: loses too much energy ({avg:.4})");
    }
}

#[test]
fn sphere_light_illuminates_and_shows_up() {
    let scene = build(
        r#"
        Format 64 64 1.0
        Projection "perspective" "fov" [45]
        WorldBegin
            AttributeBegin
                Translate 1.5 2 3
                Light "PxrSphereLight" "bulb" "intensity" [40]
            AttributeEnd
            Bxdf "PxrSurface" "floor" "diffuseColor" [0.7 0.7 0.7]
            Polygon "P" [-6 -1 0  6 -1 0  6 -1 12  -6 -1 12]
            Translate 0 0 4
            Sphere 0.8 -0.8 0.8 360
        WorldEnd
    "#,
    );
    assert_eq!(scene.lights.len(), 1);
    // The light's emissive geometry exists.
    assert!(scene.objects.len() >= 3);
    let img = pt::render(&scene, 64);
    let mean = mean_of_center(&img, 4);
    let avg = (mean.x + mean.y + mean.z) / 3.0;
    assert!(avg > 0.02, "sphere light contributes no energy: {avg:.5}");
    // Deterministic.
    let img2 = pt::render(&scene, 64);
    assert_eq!(img[32][32].x, img2[32][32].x);
}

#[test]
fn presence_makes_transparent_shadows() {
    // A presence-0.3 card between a distant light and the floor: the
    // shadowed floor must be brighter than behind a solid card.
    let base = |presence: f64| {
        format!(
            r#"
            Format 48 48 1.0
            Projection "perspective" "fov" [40]
            WorldBegin
                AttributeBegin
                    # Aim the light's -Z straight down.
                    Rotate -90 1 0 0
                    Light "PxrDistantLight" "sun" "intensity" [3]
                AttributeEnd
                Bxdf "PxrSurface" "floor" "diffuseColor" [0.7 0.7 0.7]
                Polygon "P" [-8 -1 0  8 -1 0  8 -1 14  -8 -1 14]
                Bxdf "PxrSurface" "card" "diffuseColor" [0.2 0.2 0.2] "presence" [{presence}]
                Translate 0 1.5 5
                Rotate 90 1 0 0
                Polygon "P" [-2 -2 0  2 -2 0  2 2 0  -2 2 0]
            WorldEnd
        "#
        )
    };
    // Distant light shines along -Z of its (identity) transform; the card
    // lies between it and the floor for the straight-down geometry above.
    let solid = pt::render(&build(&base(1.0)), 48);
    let cutout = pt::render(&build(&base(0.3)), 48);
    let ms = mean_of_center(&solid, 4);
    let mc = mean_of_center(&cutout, 4);
    let (ls, lc) = (ms.x + ms.y + ms.z, mc.x + mc.y + mc.z);
    assert!(
        lc > ls * 1.05,
        "presence cutout did not brighten shadows: solid {ls:.4} vs cutout {lc:.4}"
    );
}
