use std::path::PathBuf;

mod scripts;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> Result<()> {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    scripts::expand_web_types(&out_dir.join("web_types.rs"))?;
    Ok(())
}
