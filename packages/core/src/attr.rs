use crate::reactive::*;
use std::{borrow::Cow, rc::Rc};
use wasm_bindgen::{intern, JsValue};

#[derive(Clone)]
pub enum Attribute {
    Boolean(bool),
    Number(f64),
    String(&'static str),
    Shared(Rc<String>),
}

impl Attribute {
    pub fn into_js_value(self) -> JsValue {
        match self {
            Self::Boolean(t) => JsValue::from_bool(t),
            Self::Number(t) => JsValue::from_f64(t),
            Self::String(t) => JsValue::from_str(intern(t)),
            Self::Shared(t) => JsValue::from_str(&t),
        }
    }

    pub fn into_string_only(self) -> Attribute {
        match self {
            Self::Boolean(t) => intern(if t { "true" } else { "false" }).into(),
            Self::Number(t) => t.to_string().into(),
            _ => self,
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Shared(s) => s,
            Self::String(s) => intern(s),
            _ => panic!("expected a string value"),
        }
    }
}

impl From<Cow<'static, str>> for Attribute {
    fn from(t: Cow<'static, str>) -> Self {
        match t {
            Cow::Owned(t) => t.into(),
            Cow::Borrowed(t) => t.into(),
        }
    }
}

macro_rules! impl_for_types_into {
    ($($ty:ty),*) => {$(
        impl From<$ty> for Reactive<Attribute> {
            fn from(t: $ty) -> Reactive<Attribute> {
                Value(t.into())
            }
        }
    )*};
}

impl_for_types_into!(Cow<'static, str>);

macro_rules! impl_from_for_types_into {
    ($($variant:ident => $ty:ty,)*) => {$(
        impl From<$ty> for Attribute {
            fn from(t: $ty) -> Self {
                Self::$variant(t.into())
            }
        }

        impl From<$ty> for Reactive<Attribute> {
            fn from(t: $ty) -> Reactive<Attribute> {
                Value(t.into())
            }
        }
    )*};
}

impl_from_for_types_into! {
    Boolean => bool,
    Number  => f64,
    String  => &'static str,
    Shared  => String,
    Shared  => Rc<String>,
}

macro_rules! impl_for_small_nums {
    ($($ty:ident),*) => {$(
        impl From<$ty> for Attribute {
            fn from(t: $ty) -> Self {
                (t as f64).into()
            }
        }

        impl From<$ty> for Reactive<Attribute> {
            fn from(t: $ty) -> Reactive<Attribute> {
                Value(t.into())
            }
        }
    )*};
}

impl_for_small_nums!(i8, u8, i16, u16, i32, u32, i64, isize, f32);

macro_rules! impl_for_big_nums {
    ($($ty:ident),*) => {$(
        impl From<$ty> for Attribute {
            fn from(t: $ty) -> Self {
                if t < i64::MAX as $ty {
                    (t as f64).into()
                } else {
                    t.to_string().into()
                }
            }
        }

        impl From<$ty> for Reactive<Attribute> {
            fn from(t: $ty) -> Reactive<Attribute> {
                Value(t.into())
            }
        }
    )*};
}

impl_for_big_nums!(u64, i128, u128, usize);
