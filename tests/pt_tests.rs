//! Path-tracer smoke tests: deterministic, statistical, and physical
//! sanity on the Cornell box. Proper furnace/chi-square validation arrives
//! with roadmap Phase 4.

mod common;

use common::load_fixture_scene;
use render_rs::math::Vec3;
use render_rs::raytracer::pt;

fn mean_of(region: &[Vec3]) -> Vec3 {
    let n = region.len().max(1) as f64;
    region.iter().fold(Vec3::zero(), |a, b| a + *b) / n
}

#[test]
fn cornell_renders_with_gi() {
    let scene = load_fixture_scene("cornell.rib", 96, 96);
    let image = pt::render(&scene, 48);

    let all: Vec<Vec3> = image.iter().flatten().copied().collect();
    let mean = mean_of(&all);
    let lum = 0.2126 * mean.x + 0.7152 * mean.y + 0.0722 * mean.z;
    // The box is lit: neither black nor blown out.
    assert!(lum > 0.05 && lum < 2.0, "mean luminance {lum}");

    // Color bleeding: the left third (red wall side) must be redder than
    // green; the right third (green wall side) greener than red.
    let w = 96;
    let left: Vec<Vec3> = image
        .iter()
        .skip(24)
        .take(48)
        .flat_map(|row| row[..w / 3].to_vec())
        .collect();
    let right: Vec<Vec3> = image
        .iter()
        .skip(24)
        .take(48)
        .flat_map(|row| row[2 * w / 3..].to_vec())
        .collect();
    let l = mean_of(&left);
    let r = mean_of(&right);
    assert!(l.x > l.y, "left wall region not red-dominant: {l:?}");
    assert!(r.y > r.x, "right wall region not green-dominant: {r:?}");

    // Indirect illumination: the ceiling outside the light is only lit
    // indirectly (light faces down); it must be clearly non-black.
    let ceiling_corner = image[4][4];
    assert!(
        ceiling_corner.x + ceiling_corner.y + ceiling_corner.z > 0.03,
        "no indirect light on ceiling corner: {ceiling_corner:?}"
    );
}

#[test]
fn deterministic_given_seed() {
    let scene = load_fixture_scene("cornell.rib", 48, 48);
    let a = pt::render(&scene, 8);
    let b = pt::render(&scene, 8);
    for (ra, rb) in a.iter().zip(b.iter()) {
        for (pa, pb) in ra.iter().zip(rb.iter()) {
            assert!(
                pa.x == pb.x && pa.y == pb.y && pa.z == pb.z,
                "path tracer output is not deterministic"
            );
        }
    }
}

#[test]
fn converges_with_spp() {
    // Variance between two independent low-spp renders should shrink as
    // spp grows; compare 4 spp vs 64 spp against a 256-spp reference.
    let scene = load_fixture_scene("cornell.rib", 32, 32);
    let reference = pt::render(&scene, 256);
    let rmse = |img: &render_rs::output::Image| -> f64 {
        let mut sum = 0.0;
        let mut n = 0.0;
        for (r1, r2) in img.iter().zip(reference.iter()) {
            for (p1, p2) in r1.iter().zip(r2.iter()) {
                let d = *p1 - *p2;
                sum += d.dot(&d);
                n += 3.0;
            }
        }
        (sum / n).sqrt()
    };
    let coarse = rmse(&pt::render(&scene, 4));
    let fine = rmse(&pt::render(&scene, 64));
    assert!(
        fine < coarse * 0.6,
        "no convergence: rmse@4spp={coarse:.4}, rmse@64spp={fine:.4}"
    );
}
