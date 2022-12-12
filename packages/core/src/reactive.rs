use std::rc::Rc;
use xframe_reactive::{ReadSignal, Signal};

#[doc(inline)]
pub use Reactive::Value;

pub trait IntoReactive<T>: Into<Reactive<T>> {
    fn into_reactive(self) -> Reactive<T> {
        self.into()
    }

    fn into_value(self) -> T {
        let mut val = self.into_reactive();
        loop {
            match val {
                Value(t) => return t,
                Reactive::Fn(f) => val = f(),
            }
        }
    }

    fn cast<U>(self) -> Reactive<U>
    where
        T: 'static + Into<U>,
    {
        match self.into_reactive() {
            Value(t) => Value(t.into()),
            Reactive::Fn(t) => Reactive::Fn(Rc::new(move || Value(t().into_value().into()))),
        }
    }
}

impl<T, U: Into<Reactive<T>>> IntoReactive<T> for U {}

#[derive(Clone)]
pub enum Reactive<T> {
    Value(T),
    Fn(Rc<dyn Fn() -> Reactive<T>>),
}

impl<T, F, U> From<F> for Reactive<T>
where
    F: 'static + Fn() -> U,
    U: IntoReactive<T>,
{
    fn from(t: F) -> Self {
        Reactive::Fn(Rc::new(move || (t)().into()))
    }
}

impl<T, U> From<Signal<U>> for Reactive<T>
where
    U: Clone + IntoReactive<T>,
{
    fn from(t: Signal<U>) -> Self {
        Reactive::Fn(Rc::new(move || t.get().into()))
    }
}

impl<T, U> From<ReadSignal<U>> for Reactive<T>
where
    U: Clone + IntoReactive<T>,
{
    fn from(t: ReadSignal<U>) -> Self {
        Reactive::Fn(Rc::new(move || t.get().into()))
    }
}

macro_rules! impl_into_reactive {
    ($($ty:ident),*$(,)?) => {$(
        impl From<$ty> for Reactive<$ty> {
            fn from(t: $ty) -> Self {
                Value(t)
            }
        }
    )*};
}

impl_into_reactive!(
    bool, i8, u8, i16, u16, char, i32, u32, i64, u64, isize, usize, i128, u128, String
);

macro_rules! impl_into_reactive_generic {
    ($($ty:ident),*$(,)?) => {$(
        impl<T> From<$ty<T>> for Reactive<$ty<T>> {
            fn from(t: $ty<T>) -> Self {
                Value(t)
            }
        }
    )*};
}

impl_into_reactive_generic!(Rc, Option, Vec);
