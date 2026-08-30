//! Binary RIB (RISpec 3.2 Appendix C.2), roadmap Phase 9.
//!
//! Binary RIB interleaves ASCII freely with byte codes >= 0200; brackets,
//! whitespace, and comments stay ASCII. Decoding therefore translates the
//! byte stream to canonical text and hands it to the normal parser —
//! one grammar, two surface encodings. The encoder (the `render catrib
//! -binary` tool) is the inverse: requests become one-byte codes after a
//! 0314 definition, numbers become binary integers/floats, strings get
//! length-prefixed.
//!
//! Codes handled: 0200-0203 integers, 0204-0217 fixed point, 0220-0237 /
//! 0240-0243 strings, 0244 float, 0245 double, 0246 request use, 0310-0313
//! string-token definition, 0314 request definition, 0315-0316 string-token
//! reference. Reserved codes are skipped with a warning byte-by-byte.

use super::ast::{RibFile, RibValue};
use anyhow::{bail, Result};
use std::collections::HashMap;
use std::fmt::Write as _;

/// True when the buffer contains binary RIB codes (bytes >= 0200 outside
/// comments).
pub fn looks_binary(data: &[u8]) -> bool {
    let mut in_comment = false;
    for &b in data {
        match b {
            b'#' => in_comment = true,
            b'\n' => in_comment = false,
            0x80..=0xFF if !in_comment => return true,
            _ => {}
        }
    }
    false
}

fn read_be_uint(data: &[u8], pos: &mut usize, bytes: usize) -> Result<u64> {
    if *pos + bytes > data.len() {
        bail!("binary RIB: truncated integer at byte {}", *pos);
    }
    let mut v = 0u64;
    for _ in 0..bytes {
        v = (v << 8) | data[*pos] as u64;
        *pos += 1;
    }
    Ok(v)
}

fn read_be_int(data: &[u8], pos: &mut usize, bytes: usize) -> Result<i64> {
    let raw = read_be_uint(data, pos, bytes)?;
    let shift = 64 - bytes * 8;
    Ok(((raw << shift) as i64) >> shift)
}

/// Decode an encoded string that must follow (short or long form).
fn read_string(data: &[u8], pos: &mut usize) -> Result<String> {
    if *pos >= data.len() {
        bail!("binary RIB: truncated string");
    }
    let code = data[*pos];
    *pos += 1;
    let len = match code {
        0x90..=0x9F => (code - 0x90) as usize,
        0xA0..=0xA3 => read_be_uint(data, pos, (code - 0xA0) as usize + 1)? as usize,
        _ => bail!("binary RIB: expected string code, got {code:#o}"),
    };
    if *pos + len > data.len() {
        bail!("binary RIB: truncated string body");
    }
    let s = String::from_utf8_lossy(&data[*pos..*pos + len]).into_owned();
    *pos += len;
    Ok(s)
}

fn push_quoted(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Translate a (possibly) binary RIB byte stream into canonical text RIB.
pub fn decode_to_text(data: &[u8]) -> Result<String> {
    let mut out = String::with_capacity(data.len() * 2);
    let mut requests: HashMap<u8, String> = HashMap::new();
    let mut strings: HashMap<u64, String> = HashMap::new();
    let mut pos = 0usize;

    while pos < data.len() {
        let b = data[pos];
        match b {
            // ASCII passes through; comments verbatim to end of line.
            0x00..=0x7F => {
                if b == b'#' {
                    while pos < data.len() && data[pos] != b'\n' {
                        out.push(data[pos] as char);
                        pos += 1;
                    }
                } else {
                    out.push(b as char);
                    pos += 1;
                }
            }
            // Integers.
            0x80..=0x83 => {
                pos += 1;
                let v = read_be_int(data, &mut pos, (b - 0x80) as usize + 1)?;
                write!(out, " {v} ").ok();
            }
            // Fixed point: q = 4*d + w; (w+1) integer bytes, d fraction bytes.
            0x84..=0x8F => {
                pos += 1;
                let q = (b - 0x84) as usize;
                let d = q / 4;
                let w = q % 4;
                let total = w + 1 + d;
                let raw = read_be_int(data, &mut pos, total)?;
                let v = raw as f64 / 256f64.powi(d as i32);
                write!(out, " {v} ").ok();
            }
            // Strings (both forms share read_string).
            0x90..=0xA3 => {
                let s = read_string(data, &mut pos)?;
                out.push(' ');
                push_quoted(&mut out, &s);
                out.push(' ');
            }
            // Float / double, network byte order.
            0xA4 => {
                pos += 1;
                let raw = read_be_uint(data, &mut pos, 4)? as u32;
                let v = f32::from_bits(raw);
                write!(out, " {v} ").ok();
            }
            0xA5 => {
                pos += 1;
                let raw = read_be_uint(data, &mut pos, 8)?;
                let v = f64::from_bits(raw);
                write!(out, " {v} ").ok();
            }
            // Request use.
            0xA6 => {
                pos += 1;
                let code = *data
                    .get(pos)
                    .ok_or_else(|| anyhow::anyhow!("binary RIB: truncated request code"))?;
                pos += 1;
                match requests.get(&code) {
                    Some(name) => {
                        write!(out, "\n{name} ").ok();
                    }
                    None => bail!("binary RIB: request code {code} used before definition"),
                }
            }
            // Define string token.
            0xC8..=0xCB => {
                pos += 1;
                let token = read_be_uint(data, &mut pos, (b - 0xC8) as usize + 1)?;
                let s = read_string(data, &mut pos)?;
                strings.insert(token, s);
            }
            // Define request.
            0xCC => {
                pos += 1;
                let code = *data
                    .get(pos)
                    .ok_or_else(|| anyhow::anyhow!("binary RIB: truncated request def"))?;
                pos += 1;
                let name = read_string(data, &mut pos)?;
                requests.insert(code, name);
            }
            // String token reference.
            0xCD | 0xCE => {
                pos += 1;
                let token = read_be_uint(data, &mut pos, (b - 0xCD) as usize + 1)?;
                match strings.get(&token) {
                    Some(s) => {
                        out.push(' ');
                        push_quoted(&mut out, s);
                        out.push(' ');
                    }
                    None => bail!("binary RIB: string token {token} used before definition"),
                }
            }
            // Reserved / unknown: skip the byte (keeps the decoder total).
            _ => {
                pos += 1;
            }
        }
    }
    Ok(out)
}

// ---- encoder (catrib -binary) -------------------------------------------

fn put_string(out: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    if bytes.len() <= 15 {
        out.push(0x90 + bytes.len() as u8);
    } else {
        // Long form with a 2-byte length (RIB strings stay < 64K).
        out.push(0xA1);
        out.extend_from_slice(&(bytes.len() as u16).to_be_bytes());
    }
    out.extend_from_slice(bytes);
}

fn put_number(out: &mut Vec<u8>, v: f64) {
    if v.fract() == 0.0 && v.abs() < 2147483647.0 {
        let i = v as i64;
        if (-128..128).contains(&i) {
            out.push(0x80);
            out.push(i as u8);
        } else if (-32768..32768).contains(&i) {
            out.push(0x81);
            out.extend_from_slice(&(i as i16).to_be_bytes());
        } else {
            out.push(0x83);
            out.extend_from_slice(&(i as i32).to_be_bytes());
        }
        return;
    }
    // RIB numbers are single-precision (RiSpec); 0245 doubles exist but
    // catrib never needs them.
    out.push(0xA4);
    out.extend_from_slice(&(v as f32).to_bits().to_be_bytes());
}

/// Encode a parsed RIB request stream as binary RIB.
pub fn encode_binary(requests: &RibFile) -> Vec<u8> {
    let mut out = Vec::new();
    let mut codes: HashMap<&str, u8> = HashMap::new();
    let mut next_code = 0u8;

    for req in requests {
        let code = match codes.get(req.name.as_str()) {
            Some(c) => *c,
            None => {
                let c = next_code;
                next_code = next_code.wrapping_add(1);
                out.push(0xCC);
                out.push(c);
                put_string(&mut out, &req.name);
                codes.insert(req.name.as_str(), c);
                c
            }
        };
        out.push(0xA6);
        out.push(code);
        for value in &req.values {
            match value {
                RibValue::Number(n) => put_number(&mut out, *n),
                RibValue::String(s) => put_string(&mut out, s),
                RibValue::Numbers(v) => {
                    out.push(b'[');
                    for n in v {
                        put_number(&mut out, *n);
                    }
                    out.push(b']');
                }
                RibValue::Strings(v) => {
                    out.push(b'[');
                    for s in v {
                        put_string(&mut out, s);
                    }
                    out.push(b']');
                }
            }
        }
        out.push(b'\n');
    }
    out
}

/// Format a parsed RIB request stream as text (catrib's default output).
pub fn encode_text(requests: &RibFile) -> String {
    let mut out = String::new();
    for req in requests {
        out.push_str(&req.name);
        for value in &req.values {
            out.push(' ');
            match value {
                RibValue::Number(n) => {
                    write!(out, "{n}").ok();
                }
                RibValue::String(s) => push_quoted(&mut out, s),
                RibValue::Numbers(v) => {
                    out.push('[');
                    for (i, n) in v.iter().enumerate() {
                        if i > 0 {
                            out.push(' ');
                        }
                        write!(out, "{n}").ok();
                    }
                    out.push(']');
                }
                RibValue::Strings(v) => {
                    out.push('[');
                    for (i, s) in v.iter().enumerate() {
                        if i > 0 {
                            out.push(' ');
                        }
                        push_quoted(&mut out, s);
                    }
                    out.push(']');
                }
            }
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::{parse_rib, parse_rib_bytes};

    const SAMPLE: &str = r#"
        Format 320 240 1.0
        Projection "perspective" "fov" [45.5]
        WorldBegin
            Color 0.8 0.25 0.125
            Bxdf "PxrSurface" "mat" "diffuseColor" [0.5 0.25 0.75]
            Sphere 1 -1 1 360
            PointsPolygons [4] [0 1 2 3] "P" [-1 0 -1  1 0 -1  1 0 1  -1 0 1]
        WorldEnd
    "#;

    #[test]
    fn binary_round_trip() {
        let parsed = parse_rib(SAMPLE).unwrap();
        let binary = encode_binary(&parsed);
        assert!(looks_binary(&binary));
        let reparsed = parse_rib_bytes(&binary).unwrap();
        assert_eq!(parsed.len(), reparsed.len());
        for (a, b) in parsed.iter().zip(&reparsed) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.values.len(), b.values.len(), "request {}", a.name);
            for (va, vb) in a.values.iter().zip(&b.values) {
                match (va, vb) {
                    (RibValue::Number(x), RibValue::Number(y)) => {
                        assert!((x - y).abs() < 1e-6, "{x} vs {y}")
                    }
                    (RibValue::Numbers(x), RibValue::Numbers(y)) => {
                        assert_eq!(x.len(), y.len());
                        for (m, n) in x.iter().zip(y) {
                            assert!((m - n).abs() < 1e-6);
                        }
                    }
                    (a, b) => assert_eq!(a, b),
                }
            }
        }
    }

    #[test]
    fn text_round_trip_and_shrink() {
        let parsed = parse_rib(SAMPLE).unwrap();
        let text = encode_text(&parsed);
        let reparsed = parse_rib(&text).unwrap();
        assert_eq!(parsed.len(), reparsed.len());
        // Binary beats text on float-heavy content (the case that matters:
        // giant "P" arrays); tiny integer-heavy files can go either way.
        let mut heavy = String::from("PointsPolygons [4] [0 1 2 3] \"P\" [");
        for i in 0..2000 {
            heavy.push_str(&format!("{:.4} ", (i as f64 * 0.7311).sin() * 3.7));
        }
        heavy.push(']');
        let parsed_heavy = parse_rib(&heavy).unwrap();
        let text_heavy = encode_text(&parsed_heavy);
        let binary_heavy = encode_binary(&parsed_heavy);
        assert!(
            (binary_heavy.len() as f64) < text_heavy.len() as f64 * 0.75,
            "{} vs {}",
            binary_heavy.len(),
            text_heavy.len()
        );
    }

    #[test]
    fn string_token_definitions_decode() {
        // 0314 def request 0 "Sphere"; 0310 def string 7 "hello";
        // 0246 0 (Sphere) 4 ints; then a string ref.
        let mut data = Vec::new();
        data.push(0xCC);
        data.push(0);
        put_string(&mut data, "Option");
        data.push(0xA6);
        data.push(0);
        data.push(0xC8);
        data.push(7);
        put_string(&mut data, "searchpath");
        data.push(0xCD);
        data.push(7);
        data.push(0xCD);
        data.push(7);
        let text = decode_to_text(&data).unwrap();
        let parsed = parse_rib(text.as_str()).unwrap();
        assert_eq!(parsed[0].name, "Option");
        assert_eq!(parsed[0].string(0), Some("searchpath"));
    }

    /// The byte decoder and both parsers must never panic on garbage.
    #[test]
    fn fuzz_no_panic() {
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut rand = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for case in 0..300 {
            let len = (rand() % 512) as usize + 1;
            let mut data = Vec::with_capacity(len);
            for _ in 0..len {
                data.push(if case % 3 == 0 {
                    // Printable-ish ASCII soup.
                    (rand() % 96 + 32) as u8
                } else {
                    (rand() % 256) as u8
                });
            }
            // Must return Ok or Err, never panic.
            let _ = parse_rib_bytes(&data);
        }
        // Truncated binary structures specifically.
        for cut in 0..20 {
            let parsed = parse_rib("Sphere 1 -1 1 360").unwrap();
            let bin = encode_binary(&parsed);
            let _ = parse_rib_bytes(&bin[..bin.len().saturating_sub(cut)]);
        }
    }
}
