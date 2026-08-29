//! Generalized RIB request model.
//!
//! A RIB file is a flat sequence of requests: an identifier followed by
//! values (numbers, strings, and bracketed arrays of either). All
//! interpretation — which values are positional arguments and which form
//! the trailing token/value parameter list — happens in the scene builder,
//! per request. This keeps the parser total: any syntactically valid RIB
//! parses, whether or not the renderer implements the request.

#[derive(Debug, Clone, PartialEq)]
pub enum RibValue {
    Number(f64),
    String(String),
    Numbers(Vec<f64>),
    Strings(Vec<String>),
}

impl RibValue {
    pub fn as_number(&self) -> Option<f64> {
        match self {
            RibValue::Number(n) => Some(*n),
            // A one-element array is accepted where a scalar is expected;
            // real-world RIB writers do this constantly ("fov" [45]).
            RibValue::Numbers(v) if v.len() == 1 => Some(v[0]),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            RibValue::String(s) => Some(s),
            RibValue::Strings(v) if v.len() == 1 => Some(&v[0]),
            _ => None,
        }
    }

    pub fn as_numbers(&self) -> Option<&[f64]> {
        match self {
            RibValue::Numbers(v) => Some(v),
            RibValue::Number(n) => Some(std::slice::from_ref(n)),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RibRequest {
    pub name: String,
    pub values: Vec<RibValue>,
}

impl RibRequest {
    /// Positional number argument at index `i`.
    pub fn number(&self, i: usize) -> Option<f64> {
        self.values.get(i).and_then(RibValue::as_number)
    }

    /// Positional string argument at index `i`.
    pub fn string(&self, i: usize) -> Option<&str> {
        self.values.get(i).and_then(RibValue::as_str)
    }

    /// Token/value parameter pairs starting at value index `start`
    /// (everything after the positional arguments). Tokens may carry
    /// inline declarations ("uniform float roughness"); lookups match on
    /// the final word.
    pub fn params_from(&self, start: usize) -> ParamList<'_> {
        let mut params = Vec::new();
        let mut i = start;
        loop {
            let Some(token) = self.values.get(i).and_then(RibValue::as_str) else {
                break;
            };
            let Some(value) = self.values.get(i + 1) else {
                break;
            };
            params.push((token, value));
            i += 2;
        }
        ParamList(params)
    }
}

/// Borrowed view of a request's token/value parameter list.
pub struct ParamList<'a>(Vec<(&'a str, &'a RibValue)>);

impl<'a> ParamList<'a> {
    /// Find a parameter by name, ignoring any inline declaration prefix
    /// ("uniform float roughness" matches "roughness").
    pub fn get(&self, name: &str) -> Option<&'a RibValue> {
        self.0
            .iter()
            .find(|(token, _)| token.split_whitespace().last() == Some(name))
            .map(|(_, v)| *v)
    }

    pub fn get_number(&self, name: &str) -> Option<f64> {
        self.get(name).and_then(RibValue::as_number)
    }

    pub fn get_numbers(&self, name: &str) -> Option<&'a [f64]> {
        self.get(name).and_then(RibValue::as_numbers)
    }

    pub fn get_string(&self, name: &str) -> Option<&'a str> {
        self.get(name).and_then(RibValue::as_str)
    }

    pub fn iter(&self) -> impl Iterator<Item = &(&'a str, &'a RibValue)> {
        self.0.iter()
    }
}

pub type RibFile = Vec<RibRequest>;
