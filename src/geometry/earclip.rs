//! Ear-clipping triangulation for GeneralPolygon (roadmap Phase 5):
//! handles concave outlines and holes (bridged into the outer loop). The
//! polygon is projected onto its dominant plane (Newell normal); loops
//! after the first are holes.

type P3 = [f64; 3];
type P2 = [f64; 2];

fn newell_normal(pts: &[P3]) -> P3 {
    let mut n = [0.0; 3];
    for i in 0..pts.len() {
        let a = pts[i];
        let b = pts[(i + 1) % pts.len()];
        n[0] += (a[1] - b[1]) * (a[2] + b[2]);
        n[1] += (a[2] - b[2]) * (a[0] + b[0]);
        n[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    n
}

fn project(pts: &[P3], axis: usize) -> Vec<P2> {
    pts.iter()
        .map(|p| match axis {
            0 => [p[1], p[2]],
            1 => [p[2], p[0]],
            _ => [p[0], p[1]],
        })
        .collect()
}

fn signed_area(pts: &[P2]) -> f64 {
    let mut a = 0.0;
    for i in 0..pts.len() {
        let p = pts[i];
        let q = pts[(i + 1) % pts.len()];
        a += p[0] * q[1] - q[0] * p[1];
    }
    a * 0.5
}

fn cross2(o: P2, a: P2, b: P2) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

fn point_in_triangle(p: P2, a: P2, b: P2, c: P2) -> bool {
    let d1 = cross2(p, a, b);
    let d2 = cross2(p, b, c);
    let d3 = cross2(p, c, a);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

fn segments_intersect(a: P2, b: P2, c: P2, d: P2) -> bool {
    let d1 = cross2(c, d, a);
    let d2 = cross2(c, d, b);
    let d3 = cross2(a, b, c);
    let d4 = cross2(a, b, d);
    ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
}

/// Triangulate loops (first = outline, rest = holes). Returns the merged
/// vertex list (3D) and triangle indices into it.
pub fn triangulate_with_holes(loops: &[Vec<P3>]) -> Option<(Vec<P3>, Vec<u32>)> {
    let normal = newell_normal(&loops[0]);
    let axis = if normal[0].abs() >= normal[1].abs() && normal[0].abs() >= normal[2].abs() {
        0
    } else if normal[1].abs() >= normal[2].abs() {
        1
    } else {
        2
    };

    // Merged ring: outer CCW, each hole (CW) spliced via a bridge.
    let mut ring3: Vec<P3> = loops[0].clone();
    let mut ring2 = project(&ring3, axis);
    if signed_area(&ring2) < 0.0 {
        ring3.reverse();
        ring2.reverse();
    }

    for hole in &loops[1..] {
        if hole.len() < 3 {
            continue;
        }
        let mut h3 = hole.clone();
        let mut h2 = project(&h3, axis);
        if signed_area(&h2) > 0.0 {
            h3.reverse();
            h2.reverse();
        }
        // Bridge: connect the hole's rightmost vertex to the nearest ring
        // vertex whose connecting segment crosses no edge of either loop.
        let (hi, _) = h2
            .iter()
            .enumerate()
            .max_by(|a, b| a.1[0].partial_cmp(&b.1[0]).unwrap())?;
        let mut candidates: Vec<usize> = (0..ring2.len()).collect();
        candidates.sort_by(|&a, &b| {
            let da = (ring2[a][0] - h2[hi][0]).powi(2) + (ring2[a][1] - h2[hi][1]).powi(2);
            let db = (ring2[b][0] - h2[hi][0]).powi(2) + (ring2[b][1] - h2[hi][1]).powi(2);
            da.partial_cmp(&db).unwrap()
        });
        let crosses = |a: P2, b: P2, ring2: &[P2], h2: &[P2]| -> bool {
            let check = |poly: &[P2]| {
                (0..poly.len()).any(|i| {
                    let c = poly[i];
                    let d = poly[(i + 1) % poly.len()];
                    segments_intersect(a, b, c, d)
                })
            };
            check(ring2) || check(h2)
        };
        let bridge = candidates
            .into_iter()
            .find(|&ri| !crosses(ring2[ri], h2[hi], &ring2, &h2))?;

        // Splice: ring[..=bridge] + hole[hi..] + hole[..=hi] + ring[bridge..]
        let mut new3 = Vec::with_capacity(ring3.len() + h3.len() + 2);
        let mut new2 = Vec::with_capacity(ring2.len() + h2.len() + 2);
        new3.extend_from_slice(&ring3[..=bridge]);
        new2.extend_from_slice(&ring2[..=bridge]);
        for k in 0..=h3.len() {
            let idx = (hi + k) % h3.len();
            new3.push(h3[idx]);
            new2.push(h2[idx]);
        }
        new3.extend_from_slice(&ring3[bridge..]);
        new2.extend_from_slice(&ring2[bridge..]);
        ring3 = new3;
        ring2 = new2;
    }

    // Ear clipping on the merged (CCW) ring.
    let n = ring2.len();
    if n < 3 {
        return None;
    }
    let mut remaining: Vec<usize> = (0..n).collect();
    let mut indices = Vec::with_capacity((n - 2) * 3);
    let mut guard = 0usize;
    while remaining.len() > 3 && guard < n * n {
        guard += 1;
        let m = remaining.len();
        let mut clipped = false;
        for i in 0..m {
            let ia = remaining[(i + m - 1) % m];
            let ib = remaining[i];
            let ic = remaining[(i + 1) % m];
            let (a, b, c) = (ring2[ia], ring2[ib], ring2[ic]);
            if cross2(a, b, c) <= 1e-12 {
                continue; // reflex or degenerate
            }
            // Bridged rings carry coordinate-duplicate vertices; points
            // coincident with a triangle corner never block the ear.
            let same = |p: P2, q: P2| (p[0] - q[0]).abs() < 1e-12 && (p[1] - q[1]).abs() < 1e-12;
            let contains_other = remaining.iter().any(|&j| {
                if j == ia || j == ib || j == ic {
                    return false;
                }
                let p = ring2[j];
                if same(p, a) || same(p, b) || same(p, c) {
                    return false;
                }
                point_in_triangle(p, a, b, c)
            });
            if contains_other {
                continue;
            }
            indices.extend_from_slice(&[ia as u32, ib as u32, ic as u32]);
            remaining.remove(i);
            clipped = true;
            break;
        }
        if !clipped {
            // Fallback for numerically degenerate input: fan what's left.
            break;
        }
    }
    if remaining.len() >= 3 {
        for i in 1..remaining.len() - 1 {
            indices.extend_from_slice(&[
                remaining[0] as u32,
                remaining[i] as u32,
                remaining[i + 1] as u32,
            ]);
        }
    }
    Some((ring3, indices))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn area_of(positions: &[P3], indices: &[u32]) -> f64 {
        indices
            .chunks_exact(3)
            .map(|t| {
                let a = positions[t[0] as usize];
                let b = positions[t[1] as usize];
                let c = positions[t[2] as usize];
                let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let n = [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ];
                0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
            })
            .sum()
    }

    #[test]
    fn concave_polygon() {
        // L-shape, area 3.
        let outline = vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
        ];
        let (pos, idx) = triangulate_with_holes(&[outline]).unwrap();
        assert_eq!(idx.len() / 3, 4);
        assert!((area_of(&pos, &idx) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn square_with_hole() {
        // 4x4 square minus centered 2x2 hole: area 12.
        let outer = vec![
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 4.0, 0.0],
            [0.0, 4.0, 0.0],
        ];
        let hole = vec![
            [1.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
            [3.0, 3.0, 0.0],
            [1.0, 3.0, 0.0],
        ];
        let (pos, idx) = triangulate_with_holes(&[outer, hole]).unwrap();
        assert!((area_of(&pos, &idx) - 12.0).abs() < 1e-6, "area {}", area_of(&pos, &idx));
    }
}
