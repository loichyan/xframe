use std::{borrow::Cow, rc::Rc};
use wasm_bindgen::{intern, JsValue};
use xframe_reactive::{ReadSignal, Signal};

#[derive(Clone)]
pub enum Attribute {
    Boolean(bool),
    Number(f64),
    String(&'static str),
    Shared(Rc<String>),
    Reactive(Rc<dyn Fn() -> Attribute>),
}

impl Attribute {
    pub fn to_js_value(&self) -> JsValue {
        let mut val = self.clone();
        while let Self::Reactive(f) = val {
            val = f();
        }
        match val {
            Self::Boolean(t) => JsValue::from_bool(t),
            Self::Number(t) => JsValue::from_f64(t),
            Self::String(t) => JsValue::from_str(t),
            Self::Shared(t) => JsValue::from_str(&t),
            Self::Reactive(_) => unreachable!(),
        }
    }

    pub fn to_string(&self) -> Attribute {
        let mut val = self.clone();
        while let Self::Reactive(f) = val {
            val = f()
        }
        match val {
            Self::Boolean(t) => intern(if t { "true" } else { "false" }).into_attribute(),
            Self::Number(t) => {
                if t == 0.0 {
                    intern("0").into_attribute()
                } else if t == 1.0 {
                    intern("1").into_attribute()
                } else {
                    t.to_string().into_attribute()
                }
            }
            Self::String(s) => s.into_attribute(),
            Self::Shared(s) => s.into_attribute(),
            Self::Reactive(_) => unreachable!(),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Shared(s) => s,
            Self::String(s) => s,
            _ => panic!("expected a string value"),
        }
    }

    pub fn from_literal(literal: &'static str) -> Self {
        intern(literal).into_attribute()
    }
}

pub trait IntoAttribute: 'static {
    fn into_attribute(self) -> Attribute;
}

impl IntoAttribute for Attribute {
    fn into_attribute(self) -> Attribute {
        self
    }
}

impl IntoAttribute for Cow<'static, str> {
    fn into_attribute(self) -> Attribute {
        match self {
            Cow::Borrowed(s) => Attribute::String(s),
            Cow::Owned(s) => Attribute::Shared(Rc::new(s)),
        }
    }
}

impl IntoAttribute for String {
    fn into_attribute(self) -> Attribute {
        Attribute::Shared(Rc::new(self))
    }
}

impl<F, U> IntoAttribute for F
where
    F: 'static + Fn() -> U,
    U: IntoAttribute,
{
    fn into_attribute(self) -> Attribute {
        Attribute::Reactive(Rc::new(move || (self)().into_attribute()))
    }
}

impl<T> IntoAttribute for Signal<T>
where
    T: Clone + IntoAttribute,
{
    fn into_attribute(self) -> Attribute {
        Attribute::Reactive(Rc::new(move || self.get().into_attribute()))
    }
}

impl<T> IntoAttribute for ReadSignal<T>
where
    T: Clone + IntoAttribute,
{
    fn into_attribute(self) -> Attribute {
        Attribute::Reactive(Rc::new(move || self.get().into_attribute()))
    }
}

impl IntoAttribute for bool {
    fn into_attribute(self) -> Attribute {
        Attribute::Boolean(self)
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
    String => &'static str,
    Shared => Rc<String>,
}

macro_rules! impl_for_small_nums {
    ($($ty:ident),*) => {$(
        impl IntoAttribute for $ty {
            fn into_attribute(self) -> Attribute {
                Attribute::Number(self as f64)
            }
        }
    )*};
}

impl_for_small_nums!(i8, u8, i16, u16, i32, u32, i64, isize, f32);

macro_rules! impl_for_big_nums {
    ($($ty:ident),*) => {$(
        impl IntoAttribute for $ty {
            fn into_attribute(self) -> Attribute {
                if self < i64::MAX as $ty {
                    Attribute::Number(self as f64)
                } else {
                    self.to_string().into_attribute()
                }
            }
        }
    )*};
}

impl_for_big_nums!(u64, i128, u128, usize);
