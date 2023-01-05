use crate::CowStr;
use std::borrow::Cow;

#[derive(Clone)]
pub enum StringLike {
    Boolean(bool),
    Integer(i64),
    Number(f64),
    Literal(&'static str),
    String(String),
}

impl StringLike {
    pub fn into_string(self) -> CowStr {
        match self {
            Self::Boolean(t) => if t { "true" } else { "false" }.into(),
            Self::Integer(t) => match t {
                0 => "0".into(),
                1 => "1".into(),
                _ => t.to_string().into(),
            },
            Self::Number(t) => t.to_string().into(),
            Self::Literal(t) => t.into(),
            Self::String(t) => t.into(),
        }
    }
}

impl From<Cow<'static, str>> for StringLike {
    fn from(t: Cow<'static, str>) -> Self {
        match t {
            Cow::Owned(t) => t.into(),
            Cow::Borrowed(t) => t.into(),
        }
    }
}

macro_rules! impl_from_for_inner_types {
    ($($variant:ident => $ty:ty,)*) => {$(
        impl From<$ty> for StringLike {
            fn from(t: $ty) -> Self {
                Self::$variant(t)
            }
        }
    )*};
}

impl_from_for_inner_types! {
    Boolean => bool,
    Integer => i64,
    Number  => f64,
    Literal => &'static str,
    String  => String,
}

macro_rules! impl_for_small_nums {
    ($($ty:ident),*) => {$(
        impl From<$ty> for StringLike {
            fn from(t: $ty) -> Self {
                (t as i64).into()
            }
        }
    )*};
}

impl_for_small_nums!(i8, u8, i16, u16, i32, u32, isize);

impl From<f32> for StringLike {
    fn from(t: f32) -> Self {
        (t as f64).into()
    }
}

macro_rules! impl_for_big_nums {
    ($($ty:ident),*) => {$(
        impl From<$ty> for StringLike {
            fn from(t: $ty) -> Self {
                if t < i64::MAX as $ty {
                    (t as i64).into()
                } else {
                    t.to_string().into()
                }
            }
        }
    )*};
}

impl_for_big_nums!(u64, i128, u128, usize);
