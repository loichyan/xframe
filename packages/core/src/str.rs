use std::{borrow::Cow, ops::Deref, rc::Rc};

#[derive(Debug, Clone)]
pub enum RcStr {
    Literal(&'static str),
    Rc(Rc<str>),
}

impl From<&'static str> for RcStr {
    fn from(t: &'static str) -> Self {
        RcStr::Literal(t)
    }
}

impl From<Rc<str>> for RcStr {
    fn from(t: Rc<str>) -> Self {
        RcStr::Rc(t)
    }
}

impl From<Box<str>> for RcStr {
    fn from(t: Box<str>) -> Self {
        RcStr::Rc(t.into())
    }
}

impl From<String> for RcStr {
    fn from(t: String) -> Self {
        t.into_boxed_str().into()
    }
}

impl From<Cow<'static, str>> for RcStr {
    fn from(t: Cow<'static, str>) -> Self {
        match t {
            Cow::Owned(t) => t.into(),
            Cow::Borrowed(t) => t.into(),
        }
    }
}

impl Deref for RcStr {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        match self {
            RcStr::Literal(s) => s,
            RcStr::Rc(s) => s,
        }
    }
}

#[derive(Clone)]
pub enum StringLike {
    Boolean(bool),
    Integer(i64),
    Number(f64),
    Literal(&'static str),
    String(Rc<str>),
}

impl StringLike {
    pub fn into_string(self) -> RcStr {
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

impl From<RcStr> for StringLike {
    fn from(t: RcStr) -> Self {
        match t {
            RcStr::Literal(s) => s.into(),
            RcStr::Rc(s) => s.into(),
        }
    }
}

impl From<Cow<'static, str>> for StringLike {
    fn from(t: Cow<'static, str>) -> Self {
        RcStr::from(t).into()
    }
}

macro_rules! impl_from_for_inner_types {
    ($($variant:ident => $ty:ty,)*) => {$(
        impl From<$ty> for StringLike {
            fn from(t: $ty) -> Self {
                Self::$variant(t.into())
            }
        }
    )*};
}

impl_from_for_inner_types! {
    Boolean => bool,
    Integer => i64,
    Number  => f64,
    Literal => &'static str,
    String  => Rc<str>,
    String  => Box<str>,
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
