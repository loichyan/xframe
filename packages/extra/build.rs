use std::path::PathBuf;

mod codegen;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> Result<()> {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    codegen::expand_web_types(&out_dir.join("xframe_extra.rs"))?;
    Ok(())
}
