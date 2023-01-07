mod lexer;
mod parser;

#[cfg(feature = "preset")]
pub static PRESET: &str = include_str!("preset.txt");

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Element<'a> {
    pub name: &'a str,
    pub js_class: &'a str,
    pub attributes: Vec<Property<'a>>,
    pub events: Vec<Event<'a>>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Property<'a> {
    pub name: &'a str,
    pub js_type: JsType<'a>,
    pub attribute: bool,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum JsType<'a> {
    Type(&'a str),
    Literals(Literals<'a>),
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Literals<'a> {
    pub values: Vec<&'a str>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Event<'a> {
    pub name: &'a str,
    pub js_class: &'a str,
}

pub fn load(s: &str) -> Result<Vec<Element>, String> {
    parser::Parser::new(s).parse()
}

#[cfg(feature = "preset")]
pub fn load_preset() -> Vec<Element<'static>> {
    load(PRESET).unwrap()
}

#[cfg(feature = "preset")]
#[test]
fn test_preset() {
    load_preset();
}
