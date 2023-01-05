use crate::CowStr;
use std::borrow::Cow;

// TODO: rename to `StringLike`
#[derive(Clone)]
pub enum Attribute {
    Boolean(bool),
    // TODO: Add integer
    Number(f64),
    Literal(&'static str),
    String(String),
}

impl Attribute {
    pub fn into_string(self) -> CowStr {
        match self {
            Self::Boolean(t) => if t { "true" } else { "false" }.into(),
            Self::Number(t) => t.to_string().into(),
            Self::Literal(t) => t.into(),
            Self::String(t) => t.into(),
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

macro_rules! impl_from_for_types_into {
    ($($variant:ident => $ty:ty,)*) => {$(
        impl From<$ty> for Attribute {
            fn from(t: $ty) -> Self {
                Self::$variant(t.into())
            }
        }
    )*};
}

impl_from_for_types_into! {
    Boolean => bool,
    Number  => f64,
    Literal => &'static str,
    String  => String,
}

macro_rules! impl_for_small_nums {
    ($($ty:ident),*) => {$(
        impl From<$ty> for Attribute {
            fn from(t: $ty) -> Self {
                (t as f64).into()
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
    )*};
}

impl_for_big_nums!(u64, i128, u128, usize);
