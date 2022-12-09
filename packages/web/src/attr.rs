use std::{borrow::Cow, rc::Rc};
use wasm_bindgen::intern;
use xframe::Signal;

#[cfg(feature = "extra-attributes")]
#[doc(inline)]
pub use crate::generated::output::attr_types::*;

pub enum Attribute {
    Static(&'static str),
    Owned(String),
    Shared(Rc<String>),
    Reactive(Box<dyn Fn() -> Attribute>),
}

impl Attribute {
    pub fn read<U>(&self, f: impl FnOnce(&str) -> U) -> U {
        match self {
            Self::Static(t) => f(t),
            Self::Owned(t) => f(t),
            Self::Shared(t) => f(t),
            Self::Reactive(t) => (t)().read(f),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn from_literal(literal: &'static str) -> Self {
        intern(literal).into_attribute()
    }
}

pub trait IntoAttribute {
    fn into_attribute(self) -> Attribute;
}

impl IntoAttribute for Cow<'static, str> {
    fn into_attribute(self) -> Attribute {
        match self {
            Cow::Borrowed(s) => Attribute::Static(s),
            Cow::Owned(s) => Attribute::Owned(s),
        }
    }
}

impl<F, U> IntoAttribute for F
where
    F: 'static + Fn() -> U,
    U: IntoAttribute,
{
    fn into_attribute(self) -> Attribute {
        Attribute::Reactive(Box::new(move || (self)().into_attribute()))
    }
}

impl<T> IntoAttribute for Signal<T>
where
    T: Clone + IntoAttribute,
{
    fn into_attribute(self) -> Attribute {
        Attribute::Reactive(Box::new(move || self.get().into_attribute()))
    }
}

impl IntoAttribute for bool {
    fn into_attribute(self) -> Attribute {
        intern(match self {
            true => "true",
            false => "false",
        })
        .into_attribute()
    }
}

macro_rules! impl_for_wrapped_types {
    ($($variant:ident => $ty:ty,)*) => {$(
        impl IntoAttribute for $ty {
            fn into_attribute(self) -> Attribute {
                Attribute::$variant(self)
            }
        }
    )*};
}

impl_for_wrapped_types! {
    Static => &'static str,
    Owned  => String,
    Shared => Rc<String>,
}

macro_rules! impl_for_num_types {
    ($($ty:ident),*) => {$(
        impl IntoAttribute for $ty {
            fn into_attribute(self) -> Attribute {
                match self {
                    0 => intern("0").into_attribute(),
                    1 => intern("1").into_attribute(),
                    i => i.to_string().into_attribute(),
                }
            }
        }
    )*};
}

impl_for_num_types!(i8, u8, i16, u16, i32, i64, u32, u64, i128, u128, usize, isize);
