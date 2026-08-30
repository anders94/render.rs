//! USD ingest (roadmap P12 stretch): a pure-Rust importer for the .usda
//! text format, covering the subset a renderer needs — prim hierarchies
//! with Xform ops, Mesh/Sphere/Cube geometry, UsdPreviewSurface
//! materials, the standard lights, and the camera. Composition arcs
//! (references, payloads, variants, inherits) are out of scope and warn.
//!
//! The importer TRANSLATES to RIB requests and feeds the normal
//! SceneBuilder: USD is a front end, not a second scene pipeline. That
//! buys every existing feature (materials, lights, instancing-by-CTM,
//! textures) and makes the conversion inspectable — `render catrib
//! scene.usda out.rib` writes the translation.

use super::ast::{RibRequest, RibValue};
use anyhow::{bail, Result};
use std::collections::HashMap;

// ---- value model ---------------------------------------------------------

#[derive(Debug, Clone)]
enum UsdValue {
    Numbers(Vec<f64>),
    String(String),
    /// @asset path@
    Asset(String),
    /// </prim/path> target(s)
    Paths(Vec<String>),
    Tokens(Vec<String>),
}

#[derive(Debug, Default)]
struct Prim {
    type_name: String,
    name: String,
    attrs: HashMap<String, UsdValue>,
    children: Vec<Prim>,
}

// ---- tokenizer -----------------------------------------------------------

struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),   // includes dotted/namespaced names like xformOp:translate
    Number(f64),
    Str(String),
    Asset(String),
    Path(String),
    Punct(char), // ( ) [ ] { } , = ;
    Eof,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Self {
        Self { src: src.as_bytes(), pos: 0 }
    }

    fn skip_ws(&mut self) {
        loop {
            while self.pos < self.src.len() && (self.src[self.pos] as char).is_whitespace() {
                self.pos += 1;
            }
            if self.pos < self.src.len() && self.src[self.pos] == b'#' {
                while self.pos < self.src.len() && self.src[self.pos] != b'\n' {
                    self.pos += 1;
                }
                continue;
            }
            break;
        }
    }

    fn next(&mut self) -> Result<Tok> {
        self.skip_ws();
        if self.pos >= self.src.len() {
            return Ok(Tok::Eof);
        }
        let c = self.src[self.pos] as char;
        match c {
            '(' | ')' | '[' | ']' | '{' | '}' | ',' | '=' | ';' => {
                self.pos += 1;
                Ok(Tok::Punct(c))
            }
            '"' => {
                self.pos += 1;
                let start = self.pos;
                while self.pos < self.src.len() && self.src[self.pos] != b'"' {
                    if self.src[self.pos] == b'\\' {
                        self.pos += 1;
                    }
                    self.pos += 1;
                }
                let s = String::from_utf8_lossy(&self.src[start..self.pos]).into_owned();
                self.pos += 1;
                Ok(Tok::Str(s))
            }
            '@' => {
                self.pos += 1;
                let start = self.pos;
                while self.pos < self.src.len() && self.src[self.pos] != b'@' {
                    self.pos += 1;
                }
                let s = String::from_utf8_lossy(&self.src[start..self.pos]).into_owned();
                self.pos += 1;
                Ok(Tok::Asset(s))
            }
            '<' => {
                self.pos += 1;
                let start = self.pos;
                while self.pos < self.src.len() && self.src[self.pos] != b'>' {
                    self.pos += 1;
                }
                let s = String::from_utf8_lossy(&self.src[start..self.pos]).into_owned();
                self.pos += 1;
                Ok(Tok::Path(s))
            }
            _ if c.is_ascii_digit() || c == '-' || c == '+' || c == '.' => {
                let start = self.pos;
                self.pos += 1;
                while self.pos < self.src.len() {
                    let d = self.src[self.pos] as char;
                    if d.is_ascii_digit() || d == '.' || d == 'e' || d == 'E' || d == '-'
                        || d == '+'
                    {
                        self.pos += 1;
                    } else {
                        break;
                    }
                }
                let text = std::str::from_utf8(&self.src[start..self.pos]).unwrap_or("0");
                Ok(Tok::Number(text.parse().unwrap_or(0.0)))
            }
            _ => {
                let start = self.pos;
                while self.pos < self.src.len() {
                    let d = self.src[self.pos] as char;
                    if d.is_alphanumeric() || d == '_' || d == ':' || d == '.' {
                        self.pos += 1;
                    } else {
                        break;
                    }
                }
                if self.pos == start {
                    self.pos += 1; // skip unknown byte
                    return self.next();
                }
                Ok(Tok::Ident(
                    String::from_utf8_lossy(&self.src[start..self.pos]).into_owned(),
                ))
            }
        }
    }

    fn peek(&mut self) -> Result<Tok> {
        let save = self.pos;
        let t = self.next()?;
        self.pos = save;
        Ok(t)
    }
}

// ---- parser --------------------------------------------------------------

struct Parser<'a> {
    lex: Lexer<'a>,
    warnings: Vec<String>,
}

impl<'a> Parser<'a> {
    /// Skip a balanced (...) metadata block (the opener already consumed).
    fn skip_parens(&mut self) -> Result<()> {
        let mut depth = 1;
        loop {
            match self.lex.next()? {
                Tok::Punct('(') => depth += 1,
                Tok::Punct(')') => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(());
                    }
                }
                Tok::Eof => bail!("usda: unbalanced metadata parens"),
                _ => {}
            }
        }
    }

    /// Parse one value after '=': number/tuple/array/string/asset/path.
    fn parse_value(&mut self) -> Result<UsdValue> {
        match self.lex.peek()? {
            Tok::Punct('[') => {
                self.lex.next()?;
                let mut nums = Vec::new();
                let mut strs = Vec::new();
                let mut paths = Vec::new();
                loop {
                    match self.lex.next()? {
                        Tok::Punct(']') => break,
                        Tok::Punct(',') | Tok::Punct('(') | Tok::Punct(')') => {}
                        Tok::Number(n) => nums.push(n),
                        Tok::Str(s) => strs.push(s),
                        Tok::Path(p) => paths.push(p),
                        Tok::Asset(a) => strs.push(a),
                        Tok::Ident(_) => {}
                        Tok::Eof => bail!("usda: unterminated array"),
                        Tok::Punct(c) => bail!("usda: unexpected {c} in array"),
                    }
                }
                if !paths.is_empty() {
                    Ok(UsdValue::Paths(paths))
                } else if !strs.is_empty() {
                    Ok(UsdValue::Tokens(strs))
                } else {
                    Ok(UsdValue::Numbers(nums))
                }
            }
            Tok::Punct('(') => {
                // tuple like (0, 1, 2)
                self.lex.next()?;
                let mut nums = Vec::new();
                loop {
                    match self.lex.next()? {
                        Tok::Punct(')') => break,
                        Tok::Punct(',') => {}
                        Tok::Number(n) => nums.push(n),
                        Tok::Eof => bail!("usda: unterminated tuple"),
                        _ => {}
                    }
                }
                Ok(UsdValue::Numbers(nums))
            }
            Tok::Number(_) => {
                if let Tok::Number(n) = self.lex.next()? {
                    Ok(UsdValue::Numbers(vec![n]))
                } else {
                    unreachable!()
                }
            }
            Tok::Str(_) => {
                if let Tok::Str(s) = self.lex.next()? {
                    Ok(UsdValue::String(s))
                } else {
                    unreachable!()
                }
            }
            Tok::Asset(_) => {
                if let Tok::Asset(a) = self.lex.next()? {
                    Ok(UsdValue::Asset(a))
                } else {
                    unreachable!()
                }
            }
            Tok::Path(_) => {
                if let Tok::Path(p) = self.lex.next()? {
                    Ok(UsdValue::Paths(vec![p]))
                } else {
                    unreachable!()
                }
            }
            Tok::Ident(_) => {
                // bare token value (e.g. `token axis = Y` unquoted) or
                // `None`.
                if let Tok::Ident(i) = self.lex.next()? {
                    Ok(UsdValue::String(i))
                } else {
                    unreachable!()
                }
            }
            t => bail!("usda: unexpected value token {t:?}"),
        }
    }

    /// Parse the body of a prim (after '{') into attrs + children.
    fn parse_prim_body(&mut self, prim: &mut Prim) -> Result<()> {
        loop {
            let tok = self.lex.next()?;
            match tok {
                Tok::Punct('}') => return Ok(()),
                Tok::Eof => bail!("usda: unterminated prim body"),
                Tok::Ident(word) => {
                    match word.as_str() {
                        "def" | "over" | "class" => {
                            let child = self.parse_prim()?;
                            prim.children.push(child);
                        }
                        "rel" => {
                            // rel name = </path> (or a [list])
                            let name = match self.lex.next()? {
                                Tok::Ident(n) => n,
                                t => bail!("usda: rel name, got {t:?}"),
                            };
                            if let Tok::Punct('=') = self.lex.peek()? {
                                self.lex.next()?;
                                let v = self.parse_value()?;
                                prim.attrs.insert(name, v);
                            }
                        }
                        _ => {
                            // Attribute: [uniform/custom/varying]* type name
                            // [= value] [(metadata)]
                            // `word` may be a qualifier or the type; scan
                            // idents until we hit one followed by '=' ,
                            // '(' or end-of-decl — the LAST ident before
                            // '=' is the attribute name.
                            let mut last = word;
                            loop {
                                match self.lex.peek()? {
                                    Tok::Ident(_) => {
                                        if let Tok::Ident(n) = self.lex.next()? {
                                            last = n;
                                        }
                                    }
                                    _ => break,
                                }
                            }
                            match self.lex.peek()? {
                                Tok::Punct('=') => {
                                    self.lex.next()?;
                                    let v = self.parse_value()?;
                                    // trailing metadata parens?
                                    if let Tok::Punct('(') = self.lex.peek()? {
                                        self.lex.next()?;
                                        self.skip_parens()?;
                                    }
                                    prim.attrs.insert(last, v);
                                }
                                Tok::Punct('(') => {
                                    self.lex.next()?;
                                    self.skip_parens()?;
                                }
                                _ => {
                                    // declaration without value: ignore
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Parse `def Type "Name" (meta) { ... }` (the def/over/class keyword
    /// already consumed).
    fn parse_prim(&mut self) -> Result<Prim> {
        let mut prim = Prim::default();
        // Optional type name.
        if let Tok::Ident(t) = self.lex.peek()? {
            self.lex.next()?;
            prim.type_name = t;
        }
        match self.lex.next()? {
            Tok::Str(name) => prim.name = name,
            t => bail!("usda: prim name expected, got {t:?}"),
        }
        if let Tok::Punct('(') = self.lex.peek()? {
            self.lex.next()?;
            self.skip_parens()?;
            self.warnings.push(format!(
                "prim {}: composition metadata ignored (references/variants unsupported)",
                prim.name
            ));
        }
        match self.lex.next()? {
            Tok::Punct('{') => {}
            t => bail!("usda: expected {{ for prim {}, got {t:?}", prim.name),
        }
        self.parse_prim_body(&mut prim)?;
        Ok(prim)
    }

    fn parse_stage(mut self) -> Result<(Vec<Prim>, HashMap<String, UsdValue>, Vec<String>)> {
        let mut roots = Vec::new();
        let mut stage_meta = HashMap::new();
        // Optional leading stage metadata block.
        if let Tok::Punct('(') = self.lex.peek()? {
            self.lex.next()?;
            // Parse simple `key = value` pairs; skip the rest.
            let mut depth = 1;
            while depth > 0 {
                match self.lex.next()? {
                    Tok::Punct('(') => depth += 1,
                    Tok::Punct(')') => depth -= 1,
                    Tok::Ident(key) if depth == 1 => {
                        if let Tok::Punct('=') = self.lex.peek()? {
                            self.lex.next()?;
                            if let Ok(v) = self.parse_value() {
                                stage_meta.insert(key, v);
                            }
                        }
                    }
                    Tok::Eof => bail!("usda: unterminated stage metadata"),
                    _ => {}
                }
            }
        }
        loop {
            match self.lex.next()? {
                Tok::Eof => break,
                Tok::Ident(w) if w == "def" || w == "over" || w == "class" => {
                    roots.push(self.parse_prim()?);
                }
                _ => {}
            }
        }
        Ok((roots, stage_meta, self.warnings))
    }
}

// ---- conversion to RIB requests -----------------------------------------

fn req(name: &str, values: Vec<RibValue>) -> RibRequest {
    RibRequest { name: name.to_string(), values }
}

fn nums(v: &[f64]) -> RibValue {
    RibValue::Numbers(v.to_vec())
}

struct Converter<'a> {
    roots: &'a [Prim],
    out: Vec<RibRequest>,
    warnings: Vec<String>,
    z_up: bool,
}

impl<'a> Converter<'a> {
    fn attr_nums(prim: &Prim, name: &str) -> Option<Vec<f64>> {
        match prim.attrs.get(name) {
            Some(UsdValue::Numbers(v)) => Some(v.clone()),
            _ => None,
        }
    }

    fn attr_num(prim: &Prim, name: &str) -> Option<f64> {
        Self::attr_nums(prim, name).and_then(|v| v.first().copied())
    }

    /// Find a prim by absolute path like /World/Materials/Red.
    fn find_prim(&self, path: &str) -> Option<&'a Prim> {
        let mut parts = path.trim_start_matches('/').split('/');
        let first = parts.next()?;
        let mut cur = self.roots.iter().find(|p| p.name == first)?;
        for part in parts {
            cur = cur.children.iter().find(|p| p.name == part)?;
        }
        Some(cur)
    }

    /// Emit ConcatTransform requests for a prim's xformOps (in authored
    /// order).
    fn emit_xform(&mut self, prim: &Prim) {
        let order: Vec<String> = match prim.attrs.get("xformOpOrder") {
            Some(UsdValue::Tokens(t)) => t.clone(),
            _ => {
                // No explicit order: apply known ops in the canonical TRS
                // order if present.
                ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
                    .iter()
                    .filter(|k| prim.attrs.contains_key(**k))
                    .map(|s| s.to_string())
                    .collect()
            }
        };
        for op in order {
            let base = op.trim_start_matches('!'); // resetXformStack unsupported
            let vals = match prim.attrs.get(base) {
                Some(UsdValue::Numbers(v)) => v.clone(),
                _ => continue,
            };
            if base.starts_with("xformOp:translate") && vals.len() >= 3 {
                self.out.push(req("Translate", vec![
                    RibValue::Number(vals[0]),
                    RibValue::Number(vals[1]),
                    RibValue::Number(vals[2]),
                ]));
            } else if base.starts_with("xformOp:rotateXYZ") && vals.len() >= 3 {
                // RIB applies the LAST Rotate first; USD rotateXYZ means
                // Rx then Ry then Rz applied to points => emit Z, Y, X.
                self.out.push(req("Rotate", vec![
                    RibValue::Number(vals[2]),
                    RibValue::Number(0.0),
                    RibValue::Number(0.0),
                    RibValue::Number(1.0),
                ]));
                self.out.push(req("Rotate", vec![
                    RibValue::Number(vals[1]),
                    RibValue::Number(0.0),
                    RibValue::Number(1.0),
                    RibValue::Number(0.0),
                ]));
                self.out.push(req("Rotate", vec![
                    RibValue::Number(vals[0]),
                    RibValue::Number(1.0),
                    RibValue::Number(0.0),
                    RibValue::Number(0.0),
                ]));
            } else if base.starts_with("xformOp:rotateX") && vals.len() >= 1 {
                self.out.push(req("Rotate", vec![
                    RibValue::Number(vals[0]),
                    RibValue::Number(1.0),
                    RibValue::Number(0.0),
                    RibValue::Number(0.0),
                ]));
            } else if base.starts_with("xformOp:rotateY") && vals.len() >= 1 {
                self.out.push(req("Rotate", vec![
                    RibValue::Number(vals[0]),
                    RibValue::Number(0.0),
                    RibValue::Number(1.0),
                    RibValue::Number(0.0),
                ]));
            } else if base.starts_with("xformOp:rotateZ") && vals.len() >= 1 {
                self.out.push(req("Rotate", vec![
                    RibValue::Number(vals[0]),
                    RibValue::Number(0.0),
                    RibValue::Number(0.0),
                    RibValue::Number(1.0),
                ]));
            } else if base.starts_with("xformOp:scale") && vals.len() >= 3 {
                self.out.push(req("Scale", vec![
                    RibValue::Number(vals[0]),
                    RibValue::Number(vals[1]),
                    RibValue::Number(vals[2]),
                ]));
            } else if base.starts_with("xformOp:transform") && vals.len() >= 16 {
                self.out.push(req("ConcatTransform", vec![nums(&vals)]));
            } else {
                self.warnings.push(format!("unsupported xform op {base}"));
            }
        }
    }

    /// UsdPreviewSurface bound to a prim -> Bxdf request.
    fn emit_material(&mut self, prim: &Prim) {
        let shader = prim
            .attrs
            .get("material:binding")
            .and_then(|v| match v {
                UsdValue::Paths(p) => p.first().cloned(),
                _ => None,
            })
            .and_then(|path| self.find_prim(&path))
            .and_then(|mat| {
                // The surface shader is a child Shader prim with
                // info:id = UsdPreviewSurface.
                mat.children.iter().find(|c| {
                    matches!(c.attrs.get("info:id"),
                        Some(UsdValue::String(s)) if s == "UsdPreviewSurface")
                })
            });
        let Some(shader) = shader else { return };

        let diffuse =
            Self::attr_nums(shader, "inputs:diffuseColor").unwrap_or(vec![0.18, 0.18, 0.18]);
        let roughness = Self::attr_num(shader, "inputs:roughness").unwrap_or(0.5);
        let metallic = Self::attr_num(shader, "inputs:metallic").unwrap_or(0.0);
        let ior = Self::attr_num(shader, "inputs:ior").unwrap_or(1.5);
        let emissive =
            Self::attr_nums(shader, "inputs:emissiveColor").unwrap_or(vec![0.0, 0.0, 0.0]);
        let opacity = Self::attr_num(shader, "inputs:opacity").unwrap_or(1.0);

        let mut values = vec![
            RibValue::String("PxrSurface".into()),
            RibValue::String("usd".into()),
        ];
        let mut param = |name: &str, v: Vec<f64>| {
            values.push(RibValue::String(name.into()));
            values.push(RibValue::Numbers(v));
        };
        if metallic > 0.5 {
            // Metal: specular face color takes the base color.
            param("diffuseGain", vec![0.0]);
            param("specularFaceColor", diffuse.clone());
        } else {
            param("diffuseGain", vec![1.0]);
            param("diffuseColor", diffuse.clone());
            param("specularIor", vec![ior]);
        }
        param("specularRoughness", vec![roughness.max(0.02)]);
        if emissive.iter().any(|c| *c > 0.0) {
            param("glowGain", vec![1.0]);
            param("glowColor", emissive);
        }
        if opacity < 1.0 {
            param("presence", vec![opacity]);
        }
        self.out.push(req("Bxdf", values));
    }

    fn emit_light(&mut self, prim: &Prim) {
        let intensity = Self::attr_num(prim, "inputs:intensity").unwrap_or(1.0)
            * Self::attr_num(prim, "inputs:exposure").map(|e| 2f64.powf(e)).unwrap_or(1.0);
        let color = Self::attr_nums(prim, "inputs:color").unwrap_or(vec![1.0, 1.0, 1.0]);
        let mut values = |kind: &str| {
            vec![
                RibValue::String(kind.into()),
                RibValue::String(prim.name.clone()),
                RibValue::String("intensity".into()),
                RibValue::Numbers(vec![intensity]),
                RibValue::String("lightColor".into()),
                RibValue::Numbers(color.clone()),
            ]
        };
        match prim.type_name.as_str() {
            "DistantLight" => {
                let mut v = values("PxrDistantLight");
                if let Some(angle) = Self::attr_num(prim, "inputs:angle") {
                    v.push(RibValue::String("angleExtent".into()));
                    v.push(RibValue::Numbers(vec![angle]));
                }
                self.out.push(req("Light", v));
            }
            "DomeLight" => {
                let mut v = values("PxrDomeLight");
                if let Some(UsdValue::Asset(file)) = prim.attrs.get("inputs:texture:file") {
                    v.push(RibValue::String("lightColorMap".into()));
                    v.push(RibValue::String(file.clone()));
                }
                self.out.push(req("Light", v));
            }
            "SphereLight" => {
                let mut v = values("PxrSphereLight");
                if let Some(r) = Self::attr_num(prim, "inputs:radius") {
                    v.push(RibValue::String("radius".into()));
                    v.push(RibValue::Numbers(vec![r]));
                }
                self.out.push(req("Light", v));
            }
            "RectLight" => {
                let mut v = values("PxrRectLight");
                if let Some(w) = Self::attr_num(prim, "inputs:width") {
                    v.push(RibValue::String("width".into()));
                    v.push(RibValue::Numbers(vec![w]));
                }
                if let Some(h) = Self::attr_num(prim, "inputs:height") {
                    v.push(RibValue::String("height".into()));
                    v.push(RibValue::Numbers(vec![h]));
                }
                self.out.push(req("Light", v));
            }
            _ => {}
        }
    }

    fn emit_geometry(&mut self, prim: &Prim) {
        match prim.type_name.as_str() {
            "Mesh" => {
                let (Some(counts), Some(indices), Some(points)) = (
                    Self::attr_nums(prim, "faceVertexCounts"),
                    Self::attr_nums(prim, "faceVertexIndices"),
                    Self::attr_nums(prim, "points"),
                ) else {
                    self.warnings
                        .push(format!("Mesh {} missing topology; skipped", prim.name));
                    return;
                };
                let mut values = vec![nums(&counts), nums(&indices)];
                values.push(RibValue::String("P".into()));
                values.push(nums(&points));
                if let Some(n) = Self::attr_nums(prim, "normals") {
                    if n.len() == points.len() {
                        values.push(RibValue::String("N".into()));
                        values.push(nums(&n));
                    }
                }
                if let Some(st) = Self::attr_nums(prim, "primvars:st") {
                    if st.len() * 3 == points.len() * 2 {
                        values.push(RibValue::String("st".into()));
                        values.push(nums(&st));
                    }
                }
                self.out.push(req("PointsPolygons", values));
            }
            "Sphere" => {
                let r = Self::attr_num(prim, "radius").unwrap_or(1.0);
                self.out.push(req("Sphere", vec![
                    RibValue::Number(r),
                    RibValue::Number(-r),
                    RibValue::Number(r),
                    RibValue::Number(360.0),
                ]));
            }
            "Cube" => {
                let s = Self::attr_num(prim, "size").unwrap_or(2.0) / 2.0;
                let p = [
                    -s, -s, -s, s, -s, -s, s, s, -s, -s, s, -s,
                    -s, -s, s, s, -s, s, s, s, s, -s, s, s,
                ];
                self.out.push(req("PointsPolygons", vec![
                    nums(&[4.0; 6]),
                    nums(&[
                        0.0, 3.0, 2.0, 1.0, 4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 5.0, 4.0,
                        2.0, 3.0, 7.0, 6.0, 1.0, 2.0, 6.0, 5.0, 0.0, 4.0, 7.0, 3.0,
                    ]),
                    RibValue::String("P".into()),
                    nums(&p),
                ]));
            }
            _ => {}
        }
    }

    /// Depth-first prim traversal inside the world.
    fn walk(&mut self, prim: &Prim) {
        match prim.type_name.as_str() {
            "Camera" => return, // handled up front
            "Material" | "Shader" | "Scope" if prim.type_name == "Shader" => return,
            _ => {}
        }
        if prim.type_name == "Material" || prim.type_name == "Shader" {
            return;
        }
        self.out.push(req("AttributeBegin", vec![]));
        self.emit_xform(prim);
        self.emit_material(prim);
        self.emit_light(prim);
        self.emit_geometry(prim);
        for child in &prim.children {
            self.walk(child);
        }
        self.out.push(req("AttributeEnd", vec![]));
    }
}

/// Find the first camera prim and the composed transform chain to it.
fn find_camera<'p>(prims: &'p [Prim], chain: &mut Vec<&'p Prim>) -> Option<&'p Prim> {
    for p in prims {
        chain.push(p);
        if p.type_name == "Camera" {
            return Some(p);
        }
        if let Some(c) = find_camera(&p.children, chain) {
            return Some(c);
        }
        chain.pop();
    }
    None
}

/// Convert usda text into a RIB request stream.
pub fn usda_to_rib(text: &str) -> Result<Vec<RibRequest>> {
    let parser = Parser { lex: Lexer::new(text), warnings: Vec::new() };
    let (roots, meta, mut warnings) = parser.parse_stage()?;

    let z_up = matches!(meta.get("upAxis"), Some(UsdValue::String(s)) if s == "Z");
    let mut conv = Converter { roots: &roots, out: Vec::new(), warnings: Vec::new(), z_up };

    // Camera: fov from focalLength + horizontalAperture; the camera's
    // xform chain becomes the pre-WorldBegin world-to-camera transform.
    let mut chain = Vec::new();
    let camera = find_camera(&roots, &mut chain);
    let fov = camera
        .map(|c| {
            let focal = Converter::attr_num(c, "focalLength").unwrap_or(50.0);
            let aperture = Converter::attr_num(c, "horizontalAperture").unwrap_or(20.955);
            2.0 * (aperture / (2.0 * focal)).atan().to_degrees()
        })
        .unwrap_or(45.0);
    conv.out.push(req("Projection", vec![
        RibValue::String("perspective".into()),
        RibValue::String("fov".into()),
        RibValue::Numbers(vec![fov]),
    ]));

    // World-to-camera: invert the camera chain by emitting the INVERSE
    // ops in reverse order, then flip to our +z-forward convention.
    conv.out.push(req("Scale", vec![
        RibValue::Number(1.0),
        RibValue::Number(1.0),
        RibValue::Number(-1.0),
    ]));
    if camera.is_some() {
        for prim in chain.iter().rev() {
            // Inverse of each op, reversed order within the prim too.
            let order: Vec<String> = match prim.attrs.get("xformOpOrder") {
                Some(UsdValue::Tokens(t)) => t.iter().rev().cloned().collect(),
                _ => ["xformOp:scale", "xformOp:rotateXYZ", "xformOp:translate"]
                    .iter()
                    .filter(|k| prim.attrs.contains_key(**k))
                    .map(|s| s.to_string())
                    .collect(),
            };
            for op in order {
                let vals = match prim.attrs.get(op.as_str()) {
                    Some(UsdValue::Numbers(v)) => v.clone(),
                    _ => continue,
                };
                if op.starts_with("xformOp:translate") && vals.len() >= 3 {
                    conv.out.push(req("Translate", vec![
                        RibValue::Number(-vals[0]),
                        RibValue::Number(-vals[1]),
                        RibValue::Number(-vals[2]),
                    ]));
                } else if op.starts_with("xformOp:rotateXYZ") && vals.len() >= 3 {
                    // inverse of Rx Ry Rz is Rz^-1 Ry^-1 Rx^-1 applied in
                    // that order => RIB emit X then Y then Z (RIB reverses).
                    conv.out.push(req("Rotate", vec![
                        RibValue::Number(-vals[0]),
                        RibValue::Number(1.0),
                        RibValue::Number(0.0),
                        RibValue::Number(0.0),
                    ]));
                    conv.out.push(req("Rotate", vec![
                        RibValue::Number(-vals[1]),
                        RibValue::Number(0.0),
                        RibValue::Number(1.0),
                        RibValue::Number(0.0),
                    ]));
                    conv.out.push(req("Rotate", vec![
                        RibValue::Number(-vals[2]),
                        RibValue::Number(0.0),
                        RibValue::Number(0.0),
                        RibValue::Number(1.0),
                    ]));
                } else if op.starts_with("xformOp:scale") && vals.len() >= 3 {
                    conv.out.push(req("Scale", vec![
                        RibValue::Number(1.0 / vals[0].max(1e-9)),
                        RibValue::Number(1.0 / vals[1].max(1e-9)),
                        RibValue::Number(1.0 / vals[2].max(1e-9)),
                    ]));
                }
            }
        }
    }
    // Z-up stages: rotate the world so +Z becomes our +Y.
    if conv.z_up {
        conv.out.push(req("Rotate", vec![
            RibValue::Number(-90.0),
            RibValue::Number(1.0),
            RibValue::Number(0.0),
            RibValue::Number(0.0),
        ]));
    }

    conv.out.push(req("WorldBegin", vec![]));
    for root in &roots {
        conv.walk(root);
    }
    conv.out.push(req("WorldEnd", vec![]));

    warnings.extend(conv.warnings);
    for w in warnings.iter().take(8) {
        eprintln!("usda warning: {w}");
    }
    Ok(conv.out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::SceneBuilder;

    const STAGE: &str = r##"#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
)

def Xform "World"
{
    def Camera "Cam"
    {
        float focalLength = 35
        float horizontalAperture = 36
        double3 xformOp:translate = (0, 1, 10)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }

    def DistantLight "Sun"
    {
        float inputs:intensity = 2.5
        color3f inputs:color = (1, 0.9, 0.8)
    }

    def Scope "Materials"
    {
        def Material "Red"
        {
            token outputs:surface.connect = </World/Materials/Red/PBR.outputs:surface>
            def Shader "PBR"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.8, 0.1, 0.05)
                float inputs:roughness = 0.35
                float inputs:metallic = 0
            }
        }
    }

    def Xform "Geo"
    {
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Mesh "Quad"
        {
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            point3f[] points = [(-5, 0, -5), (5, 0, -5), (5, 0, 5), (-5, 0, 5)]
            rel material:binding = </World/Materials/Red>
        }

        def Sphere "Ball"
        {
            double radius = 1.5
            double3 xformOp:translate = (0, 1.5, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
    }
}
"##;

    #[test]
    fn usda_parses_and_builds() {
        let requests = usda_to_rib(STAGE).unwrap();
        let scene = SceneBuilder::new().build(&requests).unwrap();
        // Quad mesh + sphere + distant light present.
        assert_eq!(scene.lights.len(), 1);
        assert_eq!(scene.instances.len(), 1, "mesh instance");
        assert_eq!(scene.objects.len(), 1, "sphere quadric");
        // Camera fov from 35mm focal / 36mm aperture ~ 54.4 deg.
        assert!((scene.camera.fov - 54.43).abs() < 0.1, "fov {}", scene.camera.fov);
        // Material carried the UsdPreviewSurface diffuse.
        let mat = &scene.materials[scene.instances[0].material_id];
        assert!((mat.pbr.diffuse_color.x - 0.8).abs() < 1e-6);
        assert!((mat.pbr.specular_roughness - 0.35).abs() < 1e-6);
        // The camera translate (0,1,10) must shift world geometry: the
        // sphere at world (0,1.5,0) lands at camera-space z = -10 flipped
        // to +10... verify the sphere is IN FRONT of the camera.
        let desc = scene.objects[0].describe();
        let center = desc.transform.transform_point(&crate::math::Point3::new(0.0, 0.0, 0.0));
        assert!(center.z > 5.0, "sphere should be in front: {center:?}");
    }
}
