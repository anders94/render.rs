mod ast;
mod parser;
mod scene_builder;

pub mod binary;

pub use ast::*;
pub use parser::parse_rib;

/// Parse RIB from raw bytes: text UTF-8/ASCII directly, binary RIB
/// (RISpec Appendix C) via the binary decoder.
pub fn parse_rib_bytes(data: &[u8]) -> anyhow::Result<RibFile> {
    if binary::looks_binary(data) {
        let text = binary::decode_to_text(data)?;
        parse_rib(&text)
    } else {
        parse_rib(&String::from_utf8_lossy(data))
    }
}
pub use scene_builder::SceneBuilder;
