//! Total RIB tokenizer: parses any syntactically valid ASCII RIB stream
//! into a flat sequence of `RibRequest`s without knowing request
//! semantics. Interpretation lives in the scene builder.

use super::ast::{RibFile, RibRequest, RibValue};
use anyhow::{anyhow, Result};
use nom::{
    branch::alt,
    bytes::complete::{take_while, take_while1},
    character::complete::{char, multispace1},
    combinator::map,
    multi::many0,
    number::complete::double,
    IResult,
};

fn comment(input: &str) -> IResult<&str, ()> {
    let (input, _) = char('#')(input)?;
    let (input, _) = take_while(|c| c != '\n')(input)?;
    Ok((input, ()))
}

fn skip_ws(input: &str) -> IResult<&str, ()> {
    map(many0(alt((map(multispace1, |_| ()), comment))), |_| ())(input)
}

fn string_literal(input: &str) -> IResult<&str, String> {
    let (input, _) = char('"')(input)?;
    let (input, s) = take_while(|c| c != '"')(input)?;
    let (input, _) = char('"')(input)?;
    Ok((input, s.to_string()))
}

fn identifier(input: &str) -> IResult<&str, &str> {
    take_while1(|c: char| c.is_ascii_alphabetic() || c == '_')(input)
}

/// Bracketed array: homogeneous numbers or strings (RIB arrays never mix).
fn array_value(input: &str) -> IResult<&str, RibValue> {
    let (input, _) = char('[')(input)?;
    let (input, _) = skip_ws(input)?;

    if input.starts_with('"') {
        let mut strings = Vec::new();
        let mut rest = input;
        loop {
            let (r, _) = skip_ws(rest)?;
            if let Ok((r, _)) = char::<_, nom::error::Error<&str>>(']')(r) {
                return Ok((r, RibValue::Strings(strings)));
            }
            let (r, s) = string_literal(r)?;
            strings.push(s);
            rest = r;
        }
    }

    let mut numbers = Vec::new();
    let mut rest = input;
    loop {
        let (r, _) = skip_ws(rest)?;
        if let Ok((r, _)) = char::<_, nom::error::Error<&str>>(']')(r) {
            return Ok((r, RibValue::Numbers(numbers)));
        }
        let (r, n) = double(r)?;
        numbers.push(n);
        rest = r;
    }
}

/// A single value: string, array, or number. Fails on an identifier
/// (which starts the next request).
fn value(input: &str) -> IResult<&str, RibValue> {
    if input.starts_with('"') {
        return map(string_literal, RibValue::String)(input);
    }
    if input.starts_with('[') {
        return array_value(input);
    }
    // Guard: `double` would happily parse the "inf" in "Infinity" or an
    // identifier like "nan..."; requests must win, so only parse numbers
    // that start numerically.
    let starts_numeric = input
        .chars()
        .next()
        .map(|c| c.is_ascii_digit() || c == '-' || c == '+' || c == '.')
        .unwrap_or(false);
    if starts_numeric {
        return map(double, RibValue::Number)(input);
    }
    Err(nom::Err::Error(nom::error::Error::new(
        input,
        nom::error::ErrorKind::Alt,
    )))
}

fn request(input: &str) -> IResult<&str, RibRequest> {
    let (input, _) = skip_ws(input)?;
    let (input, name) = identifier(input)?;
    let mut values = Vec::new();
    let mut rest = input;
    loop {
        let (r, _) = skip_ws(rest)?;
        match value(r) {
            Ok((r, v)) => {
                values.push(v);
                rest = r;
            }
            Err(_) => {
                rest = r;
                break;
            }
        }
    }
    Ok((
        rest,
        RibRequest {
            name: name.to_string(),
            values,
        },
    ))
}

fn rib_file(input: &str) -> IResult<&str, RibFile> {
    let (input, requests) = many0(request)(input)?;
    let (input, _) = skip_ws(input)?;
    Ok((input, requests))
}

pub fn parse_rib(input: &str) -> Result<RibFile> {
    match rib_file(input) {
        Ok((rest, requests)) => {
            let rest = rest.trim();
            if !rest.is_empty() {
                let preview: String = rest.chars().take(60).collect();
                return Err(anyhow!("failed to parse RIB near: {preview:?}"));
            }
            Ok(requests)
        }
        Err(e) => Err(anyhow!("failed to parse RIB file: {e:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_display() {
        let file = parse_rib(r#"Display "output.ppm" "file" "rgb""#).unwrap();
        assert_eq!(file.len(), 1);
        assert_eq!(file[0].name, "Display");
        assert_eq!(file[0].string(0), Some("output.ppm"));
    }

    #[test]
    fn test_parse_sphere() {
        let file = parse_rib("Sphere 1.0 -1.0 1.0 360").unwrap();
        assert_eq!(file[0].name, "Sphere");
        assert_eq!(file[0].number(3), Some(360.0));
    }

    #[test]
    fn test_parse_simple_rib() {
        let input = r#"
            ##RenderMan RIB-Structure 1.1
            version 3.04
            Display "output.ppm" "file" "rgb"
            Format 640 480 1.0
            WorldBegin
                Sphere 1.0 -1.0 1.0 360
            WorldEnd
        "#;
        let file = parse_rib(input).unwrap();
        let names: Vec<&str> = file.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(
            names,
            ["version", "Display", "Format", "WorldBegin", "Sphere", "WorldEnd"]
        );
    }

    #[test]
    fn test_unknown_requests_parse() {
        // Compliance policy: anything syntactically valid parses.
        let input = r#"
            Option "searchpath" "string shader" ["/shaders:@"]
            Attribute "identifier" "name" ["hero"]
            GeometricApproximation "motionfactor" 1.0
            SubdivisionMesh "catmull-clark" [4] [0 1 2 3] ["interpolateboundary"] [0 0] [] []
            ObjectBegin 1
            ObjectEnd
        "#;
        let file = parse_rib(input).unwrap();
        assert_eq!(file.len(), 6);
        assert_eq!(file[3].name, "SubdivisionMesh");
        // Empty arrays parse as empty Numbers.
        assert_eq!(file[3].values.last(), Some(&RibValue::Numbers(vec![])));
    }

    #[test]
    fn test_param_list_with_inline_declaration() {
        let file =
            parse_rib(r#"Surface "plastic" "uniform float roughness" [0.25] "Ks" 0.6"#).unwrap();
        let params = file[0].params_from(1);
        assert_eq!(params.get_number("roughness"), Some(0.25));
        assert_eq!(params.get_number("Ks"), Some(0.6));
    }

    #[test]
    fn test_negative_and_scientific_numbers() {
        let file = parse_rib("ConcatTransform [1 0 0 0 0 1 0 0 0 0 1 0 -2.5e-1 0 3 1]").unwrap();
        let m = file[0].values[0].as_numbers().unwrap();
        assert_eq!(m.len(), 16);
        assert!((m[12] + 0.25).abs() < 1e-12);
    }

    #[test]
    fn test_string_array() {
        let file = parse_rib(r#"Procedural "DelayedReadArchive" ["big.rib"] [-1 1 -1 1 -1 1]"#)
            .unwrap();
        assert_eq!(
            file[0].values[1],
            RibValue::Strings(vec!["big.rib".to_string()])
        );
    }
}
