use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
use xframe_reactive::{ReadSignal, Scope, Signal};

#[doc(inline)]
pub use Reactive::Value;

#[derive(Clone)]
pub enum Reactive<T> {
    Value(T),
    Dyn(Rc<dyn Fn() -> Reactive<T>>),
}

impl<T: 'static> Reactive<T> {
    pub fn into_value(self) -> T {
        let mut val = self;
        loop {
            match val {
                Value(t) => return t,
                Reactive::Dyn(f) => val = f(),
            }
        }
    }

    pub fn cast<U: 'static>(self) -> Reactive<U>
    where
        T: Into<U>,
    {
        self.cast_with(T::into)
    }

    pub fn cast_with<U: 'static>(self, f: fn(T) -> U) -> Reactive<U> {
        match self {
            Value(t) => Value(f(t)),
            Reactive::Dyn(t) => Reactive::Dyn(Rc::new(move || Value(f(t().into_value())))),
        }
    }
}

pub trait IntoReactive<T: 'static> {
    fn into_reactive(self, cx: Scope) -> Reactive<T>;
}

impl<T: 'static> IntoReactive<T> for Reactive<T> {
    fn into_reactive(self, _: Scope) -> Reactive<T> {
        self
    }
}

impl<T, F, U> IntoReactive<T> for F
where
    T: 'static,
    F: 'static + Fn() -> U,
    U: IntoReactive<T>,
{
    fn into_reactive(self, cx: Scope) -> Reactive<T> {
        Reactive::Dyn(Rc::new(move || self().into_reactive(cx)))
    }
}

impl<T, U> IntoReactive<T> for Signal<U>
where
    T: 'static,
    U: 'static + Clone + IntoReactive<T>,
{
    fn into_reactive(self, cx: Scope) -> Reactive<T> {
        ReadSignal::from(self).into_reactive(cx)
    }
}

impl<T, U> IntoReactive<T> for ReadSignal<U>
where
    T: 'static,
    U: 'static + Clone + IntoReactive<T>,
{
    fn into_reactive(self, cx: Scope) -> Reactive<T> {
        Reactive::Dyn(Rc::new(move || self.get().into_reactive(cx)))
    }
}

macro_rules! impl_into_reactive {
    ($($ty:ident),*$(,)?) => {$(
        impl IntoReactive<$ty> for $ty {
            fn into_reactive(self, _: Scope) -> Reactive<$ty> {
                Value(self)
            }
        }
    )*};
}

impl_into_reactive!(
    bool, i8, u8, i16, u16, char, i32, u32, i64, u64, isize, usize, i128, u128, String
);

macro_rules! impl_into_reactive_generic {
    ($($ty:ident),*$(,)?) => {$(
        impl<T: 'static> IntoReactive<$ty<T>> for $ty<T> {
            fn into_reactive(self, _: Scope) -> Reactive<$ty<T>> {
                Value(self)
            }
        }
    )*};
}

impl_into_reactive_generic!(Rc, Option, Vec, RefCell, Cell);
