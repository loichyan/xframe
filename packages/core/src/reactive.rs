use std::rc::Rc;
use xframe_reactive::{ReadSignal, Signal};

#[doc(inline)]
pub use Reactive::Value;

#[derive(Clone)]
pub enum Reactive<T: 'static> {
    Value(T),
    Fn(Rc<dyn Fn() -> Reactive<T>>),
}

impl<T> Reactive<T> {
    pub fn into_value(self) -> T {
        let mut val = self;
        loop {
            match val {
                Value(t) => return t,
                Reactive::Fn(f) => val = f(),
            }
        }
    }

    pub fn cast<U>(self) -> Reactive<U>
    where
        T: Into<U>,
    {
        match self {
            Value(t) => Value(t.into()),
            Reactive::Fn(t) => Reactive::Fn(Rc::new(move || t().cast())),
        }
    }
}

impl<T, F, U> From<F> for Reactive<T>
where
    F: 'static + Fn() -> U,
    U: Into<Reactive<T>>,
{
    fn from(t: F) -> Self {
        Reactive::Fn(Rc::new(move || (t)().into()))
    }
}

impl<T, U> From<Signal<U>> for Reactive<T>
where
    U: Clone + Into<Reactive<T>>,
{
    fn from(t: Signal<U>) -> Self {
        Reactive::Fn(Rc::new(move || t.get().into()))
    }
}

impl<T, U> From<ReadSignal<U>> for Reactive<T>
where
    U: Clone + Into<Reactive<T>>,
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
