use std::borrow::Cow;
use wasm_bindgen::intern;

pub use crate::generated::output::attr_types::*;

pub trait AsCowStr {
    fn as_cow_str(&self) -> Cow<str>;
}

impl<T: AsCowStr + ?Sized> AsCowStr for &T {
    fn as_cow_str(&self) -> Cow<str> {
        T::as_cow_str(self)
    }
}

impl AsCowStr for str {
    fn as_cow_str(&self) -> Cow<str> {
        Cow::Borrowed(self)
    }
}

impl AsCowStr for bool {
    fn as_cow_str(&self) -> Cow<str> {
        intern(match self {
            true => "true",
            false => "false",
        })
        .into()
    }
}

impl AsCowStr for i32 {
    fn as_cow_str(&self) -> Cow<str> {
        match self {
            0 => intern("0").into(),
            1 => intern("1").into(),
            i => i.to_string().into(),
        }
    }
}

pub(crate) fn cow_str_from_literal(s: &'static str) -> Cow<str> {
    intern(s).into()
}
