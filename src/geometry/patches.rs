//! Parametric patch tessellation (roadmap Phase 5): bilinear and bicubic
//! PatchMesh/Patch with RiSpec basis matrices, and NuPatch NURBS surfaces
//! evaluated with de Boor basis functions (trim curves are deferred).
//! Everything dices into the standard Mesh; grid vertices are shared
//! across patch boundaries so the tessellation is crack-free.

use crate::geometry::subdiv::smooth_normals;
use crate::geometry::Mesh;

pub type Basis4 = [[f64; 4]; 4];

pub const BEZIER: Basis4 = [
    [-1.0, 3.0, -3.0, 1.0],
    [3.0, -6.0, 3.0, 0.0],
    [-3.0, 3.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
];
pub const BSPLINE: Basis4 = [
    [-1.0 / 6.0, 3.0 / 6.0, -3.0 / 6.0, 1.0 / 6.0],
    [3.0 / 6.0, -6.0 / 6.0, 3.0 / 6.0, 0.0],
    [-3.0 / 6.0, 0.0, 3.0 / 6.0, 0.0],
    [1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0, 0.0],
];
pub const CATMULL_ROM: Basis4 = [
    [-0.5, 1.5, -1.5, 0.5],
    [1.0, -2.5, 2.0, -0.5],
    [-0.5, 0.0, 0.5, 0.0],
    [0.0, 1.0, 0.0, 0.0],
];
pub const HERMITE: Basis4 = [
    [2.0, 1.0, -2.0, 1.0],
    [-3.0, -2.0, 3.0, -1.0],
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
];
pub const POWER: Basis4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// RiSpec basis name -> (matrix, step).
pub fn basis_by_name(name: &str) -> Option<(Basis4, usize)> {
    match name {
        "bezier" => Some((BEZIER, 3)),
        "b-spline" | "bspline" => Some((BSPLINE, 1)),
        "catmull-rom" | "catmullrom" => Some((CATMULL_ROM, 1)),
        "hermite" => Some((HERMITE, 2)),
        "power" => Some((POWER, 4)),
        _ => None,
    }
}

fn basis_weights(basis: &Basis4, t: f64) -> [f64; 4] {
    let tv = [t * t * t, t * t, t, 1.0];
    let mut w = [0.0; 4];
    for (col, wc) in w.iter_mut().enumerate() {
        for row in 0..4 {
            *wc += tv[row] * basis[row][col];
        }
    }
    w
}

/// Evaluate a bicubic patch (4x4 control grid) at (u, v).
fn eval_bicubic(basis_u: &Basis4, basis_v: &Basis4, ctrl: &[[f64; 3]; 16], u: f64, v: f64) -> [f64; 3] {
    let bu = basis_weights(basis_u, u);
    let bv = basis_weights(basis_v, v);
    let mut p = [0.0; 3];
    for (j, bvj) in bv.iter().enumerate() {
        for (i, bui) in bu.iter().enumerate() {
            let c = ctrl[j * 4 + i];
            let w = bui * bvj;
            p[0] += c[0] * w;
            p[1] += c[1] * w;
            p[2] += c[2] * w;
        }
    }
    p
}

pub struct PatchMeshDef<'a> {
    /// Control points, row-major (nu columns × nv rows), xyz triples.
    pub points: &'a [f64],
    pub nu: usize,
    pub nv: usize,
    pub u_wrap: bool,
    pub v_wrap: bool,
}

/// Tessellate a bicubic PatchMesh with the given bases into a Mesh.
/// `segs` = subdivisions per patch span.
pub fn tessellate_bicubic(
    def: &PatchMeshDef<'_>,
    basis_u: &Basis4,
    ustep: usize,
    basis_v: &Basis4,
    vstep: usize,
    segs: usize,
) -> Option<Mesh> {
    let (nu, nv) = (def.nu, def.nv);
    if def.points.len() < nu * nv * 3 || nu < 4 && !def.u_wrap || nv < 4 && !def.v_wrap {
        return None;
    }
    let npu = if def.u_wrap { nu / ustep } else { (nu - 4) / ustep + 1 };
    let npv = if def.v_wrap { nv / vstep } else { (nv - 4) / vstep + 1 };
    if npu == 0 || npv == 0 {
        return None;
    }
    let ctrl = |i: usize, j: usize| -> [f64; 3] {
        let idx = ((j % nv) * nu + (i % nu)) * 3;
        [def.points[idx], def.points[idx + 1], def.points[idx + 2]]
    };

    // Shared sample grid: (npu*segs + 1) x (npv*segs + 1); wrapped
    // directions drop the duplicated seam column/row.
    let gu = npu * segs + if def.u_wrap { 0 } else { 1 };
    let gv = npv * segs + if def.v_wrap { 0 } else { 1 };
    let mut positions = Vec::with_capacity(gu * gv);
    let mut st = Vec::with_capacity(gu * gv);
    for sv in 0..gv {
        let pv = (sv / segs).min(npv - 1);
        let v = sv as f64 / segs as f64 - pv as f64;
        for su in 0..gu {
            let pu = (su / segs).min(npu - 1);
            let u = su as f64 / segs as f64 - pu as f64;
            st.push([
                (su as f64 / (npu * segs) as f64) as f32,
                (sv as f64 / (npv * segs) as f64) as f32,
            ]);
            let mut grid = [[0.0; 3]; 16];
            for (jj, row) in grid.chunks_exact_mut(4).enumerate() {
                for (ii, c) in row.iter_mut().enumerate() {
                    *c = ctrl(pu * ustep + ii, pv * vstep + jj);
                }
            }
            let p = eval_bicubic(basis_u, basis_v, &grid, u, v);
            positions.push([p[0] as f32, p[1] as f32, p[2] as f32]);
        }
    }

    let mut indices = Vec::new();
    let cols = gu;
    let wrap_u = def.u_wrap;
    let wrap_v = def.v_wrap;
    let cells_u = if wrap_u { gu } else { gu - 1 };
    let cells_v = if wrap_v { gv } else { gv - 1 };
    for cv in 0..cells_v {
        for cu in 0..cells_u {
            let a = (cv * cols + cu) as u32;
            let b = (cv * cols + (cu + 1) % gu) as u32;
            let c = (((cv + 1) % gv) * cols + (cu + 1) % gu) as u32;
            let d = (((cv + 1) % gv) * cols + cu) as u32;
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    let normals = smooth_normals(&positions, &indices);
    Some(Mesh::new(positions, indices, Some(normals), Some(st)))
}

/// Bilinear PatchMesh: control points interpolated directly.
pub fn tessellate_bilinear(def: &PatchMeshDef<'_>, segs: usize) -> Option<Mesh> {
    let (nu, nv) = (def.nu, def.nv);
    if def.points.len() < nu * nv * 3 || nu < 2 || nv < 2 {
        return None;
    }
    let npu = if def.u_wrap { nu } else { nu - 1 };
    let npv = if def.v_wrap { nv } else { nv - 1 };
    let gu = npu * segs + if def.u_wrap { 0 } else { 1 };
    let gv = npv * segs + if def.v_wrap { 0 } else { 1 };
    let ctrl = |i: usize, j: usize| -> [f64; 3] {
        let idx = ((j % nv) * nu + (i % nu)) * 3;
        [def.points[idx], def.points[idx + 1], def.points[idx + 2]]
    };
    let mut positions = Vec::with_capacity(gu * gv);
    let mut st = Vec::with_capacity(gu * gv);
    for sv in 0..gv {
        let pv = (sv / segs).min(npv - 1);
        let v = sv as f64 / segs as f64 - pv as f64;
        for su in 0..gu {
            let pu = (su / segs).min(npu - 1);
            let u = su as f64 / segs as f64 - pu as f64;
            st.push([
                (su as f64 / (npu * segs) as f64) as f32,
                (sv as f64 / (npv * segs) as f64) as f32,
            ]);
            let p00 = ctrl(pu, pv);
            let p10 = ctrl(pu + 1, pv);
            let p01 = ctrl(pu, pv + 1);
            let p11 = ctrl(pu + 1, pv + 1);
            let mut p = [0.0; 3];
            for k in 0..3 {
                let a = p00[k] * (1.0 - u) + p10[k] * u;
                let b = p01[k] * (1.0 - u) + p11[k] * u;
                p[k] = a * (1.0 - v) + b * v;
            }
            positions.push([p[0] as f32, p[1] as f32, p[2] as f32]);
        }
    }
    let mut indices = Vec::new();
    let cells_u = if def.u_wrap { gu } else { gu - 1 };
    let cells_v = if def.v_wrap { gv } else { gv - 1 };
    for cv in 0..cells_v {
        for cu in 0..cells_u {
            let a = (cv * gu + cu) as u32;
            let b = (cv * gu + (cu + 1) % gu) as u32;
            let c = (((cv + 1) % gv) * gu + (cu + 1) % gu) as u32;
            let d = (((cv + 1) % gv) * gu + cu) as u32;
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    let normals = smooth_normals(&positions, &indices);
    Some(Mesh::new(positions, indices, Some(normals), Some(st)))
}

// ---------------------------------------------------------------------------
// NURBS (NuPatch)

pub struct NuPatchDef<'a> {
    pub nu: usize,
    pub uorder: usize,
    pub uknot: &'a [f64],
    pub umin: f64,
    pub umax: f64,
    pub nv: usize,
    pub vorder: usize,
    pub vknot: &'a [f64],
    pub vmin: f64,
    pub vmax: f64,
    /// Homogeneous control points (x y z w) when rational, else xyz.
    pub points: &'a [f64],
    pub rational: bool,
}

/// All B-spline basis values of the given order at parameter t
/// (zero except within the active span); simple Cox-de Boor recursion.
fn bspline_basis(knots: &[f64], n_ctrl: usize, order: usize, t: f64) -> Vec<f64> {
    let degree = order - 1;
    let mut basis = vec![0.0; n_ctrl + degree + 1];
    // Find the knot span (clamped to valid range).
    let hi = (n_ctrl).min(knots.len().saturating_sub(order));
    let mut span = degree;
    for i in degree..hi {
        if t < knots[i + 1] || i == hi - 1 {
            span = i;
            break;
        }
        span = i + 1;
    }
    basis[span] = 1.0;
    for p in 1..=degree {
        for i in (span.saturating_sub(p)..=span).rev() {
            let mut val = 0.0;
            let d1 = knots.get(i + p).copied().unwrap_or(0.0) - knots[i];
            if d1 > 1e-12 {
                val += basis[i] * (t - knots[i]) / d1;
            }
            let d2 = knots.get(i + p + 1).copied().unwrap_or(0.0)
                - knots.get(i + 1).copied().unwrap_or(0.0);
            if d2 > 1e-12 {
                val += basis[i + 1] * (knots[i + p + 1] - t) / d2;
            }
            basis[i] = val;
        }
    }
    basis.truncate(n_ctrl);
    basis
}

fn eval_nurbs(def: &NuPatchDef<'_>, u: f64, v: f64) -> [f64; 3] {
    let bu = bspline_basis(def.uknot, def.nu, def.uorder, u);
    let bv = bspline_basis(def.vknot, def.nv, def.vorder, v);
    let stride = if def.rational { 4 } else { 3 };
    let mut acc = [0.0f64; 3];
    let mut wacc = 0.0f64;
    for (j, bvj) in bv.iter().enumerate() {
        if *bvj == 0.0 {
            continue;
        }
        for (i, bui) in bu.iter().enumerate() {
            let wgt = bui * bvj;
            if wgt == 0.0 {
                continue;
            }
            let idx = (j * def.nu + i) * stride;
            let w = if def.rational { def.points[idx + 3] } else { 1.0 };
            acc[0] += def.points[idx] * wgt * if def.rational { 1.0 } else { 1.0 };
            acc[1] += def.points[idx + 1] * wgt;
            acc[2] += def.points[idx + 2] * wgt;
            wacc += w * wgt;
        }
    }
    if def.rational && wacc.abs() > 1e-12 {
        // Homogeneous points are (wx, wy, wz, w) per RiSpec "Pw".
        [acc[0] / wacc, acc[1] / wacc, acc[2] / wacc]
    } else {
        acc
    }
}

pub fn tessellate_nurbs(def: &NuPatchDef<'_>, segs_u: usize, segs_v: usize) -> Option<Mesh> {
    let stride = if def.rational { 4 } else { 3 };
    if def.points.len() < def.nu * def.nv * stride {
        return None;
    }
    let gu = segs_u + 1;
    let gv = segs_v + 1;
    let mut positions = Vec::with_capacity(gu * gv);
    let mut st = Vec::with_capacity(gu * gv);
    for sv in 0..gv {
        let v = def.vmin + (def.vmax - def.vmin) * sv as f64 / segs_v as f64;
        for su in 0..gu {
            let u = def.umin + (def.umax - def.umin) * su as f64 / segs_u as f64;
            st.push([
                (su as f64 / segs_u as f64) as f32,
                (sv as f64 / segs_v as f64) as f32,
            ]);
            let p = eval_nurbs(def, u, v);
            positions.push([p[0] as f32, p[1] as f32, p[2] as f32]);
        }
    }
    let mut indices = Vec::new();
    for cv in 0..segs_v {
        for cu in 0..segs_u {
            let a = (cv * gu + cu) as u32;
            let b = (cv * gu + cu + 1) as u32;
            let c = ((cv + 1) * gu + cu + 1) as u32;
            let d = ((cv + 1) * gu + cu) as u32;
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    let normals = smooth_normals(&positions, &indices);
    Some(Mesh::new(positions, indices, Some(normals), Some(st)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bezier_patch_interpolates_corners() {
        // Flat 4x4 bezier sheet with a raised center.
        let mut pts = Vec::new();
        for j in 0..4 {
            for i in 0..4 {
                let z = if (1..=2).contains(&i) && (1..=2).contains(&j) { 1.0 } else { 0.0 };
                pts.extend_from_slice(&[i as f64, j as f64, z]);
            }
        }
        let def = PatchMeshDef { points: &pts, nu: 4, nv: 4, u_wrap: false, v_wrap: false };
        let mesh = tessellate_bicubic(&def, &BEZIER, 3, &BEZIER, 3, 8).unwrap();
        // Bezier interpolates its corner control points exactly.
        let p0 = mesh.positions[0];
        assert!((p0[0] - 0.0).abs() < 1e-5 && (p0[1] - 0.0).abs() < 1e-5);
        let last = *mesh.positions.last().unwrap();
        assert!((last[0] - 3.0).abs() < 1e-5 && (last[1] - 3.0).abs() < 1e-5);
        // Center bulges upward but below control cage.
        let zmax = mesh.positions.iter().map(|p| p[2]).fold(f32::MIN, f32::max);
        assert!(zmax > 0.3 && zmax < 1.0, "zmax {zmax}");
        assert_eq!(mesh.positions.len(), 81);
    }

    #[test]
    fn bspline_mesh_multiple_patches_no_cracks() {
        // 7x4 b-spline control grid -> 4x1 patches; shared-grid tessellation
        // must produce a single welded sheet.
        let mut pts = Vec::new();
        for j in 0..4 {
            for i in 0..7 {
                pts.extend_from_slice(&[i as f64, j as f64, ((i * 3 + j) % 2) as f64 * 0.3]);
            }
        }
        let def = PatchMeshDef { points: &pts, nu: 7, nv: 4, u_wrap: false, v_wrap: false };
        let mesh = tessellate_bicubic(&def, &BSPLINE, 1, &BSPLINE, 1, 4).unwrap();
        assert_eq!(mesh.positions.len(), (4 * 4 + 1) * (1 * 4 + 1));
        assert_eq!(mesh.triangle_count(), (4 * 4) * 4 * 2);
    }

    #[test]
    fn nurbs_flat_sheet() {
        // Order-3 (quadratic) NURBS plane with uniform clamped knots.
        let nu = 4;
        let nv = 3;
        let uknot = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0];
        let vknot = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut pts = Vec::new();
        for j in 0..nv {
            for i in 0..nu {
                pts.extend_from_slice(&[i as f64, j as f64, 0.0]);
            }
        }
        let def = NuPatchDef {
            nu,
            uorder: 3,
            uknot: &uknot,
            umin: 0.0,
            umax: 1.0,
            nv,
            vorder: 3,
            vknot: &vknot,
            vmin: 0.0,
            vmax: 1.0,
            points: &pts,
            rational: false,
        };
        let mesh = tessellate_nurbs(&def, 8, 8).unwrap();
        // A flat control net evaluates to the flat plane (z = 0) and stays
        // within the control hull.
        for p in &mesh.positions {
            assert!(p[2].abs() < 1e-5);
            assert!(p[0] >= -1e-5 && p[0] <= 3.0 + 1e-4);
        }
        // Clamped ends interpolate the corner control points.
        let p0 = mesh.positions[0];
        assert!(p0[0].abs() < 1e-5 && p0[1].abs() < 1e-5);
    }
}
