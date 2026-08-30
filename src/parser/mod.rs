mod ast;
mod parser;
mod scene_builder;

pub mod binary;
pub mod usd;

pub use ast::*;
pub use parser::parse_rib;

/// Parse RIB from raw bytes: text UTF-8/ASCII directly, binary RIB
/// (RISpec Appendix C) via the binary decoder.
pub fn parse_rib_bytes(data: &[u8]) -> anyhow::Result<RibFile> {
    // usda sniff: the standard magic comment.
    if data.starts_with(b"#usda") {
        return usd::usda_to_rib(&String::from_utf8_lossy(data));
    }
    if binary::looks_binary(data) {
        let text = binary::decode_to_text(data)?;
        parse_rib(&text)
    } else {
        parse_rib(&String::from_utf8_lossy(data))
    }
}
pub use scene_builder::SceneBuilder;
