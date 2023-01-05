use std::rc::Rc;
use xframe_reactive::{ReadSignal, Signal};

#[derive(Clone)]
pub enum Reactive<T> {
    Variable(T),
    Static(T),
    Fn(Rc<dyn Fn() -> T>),
}

impl<T: 'static> Reactive<T> {
    pub fn into_value(self) -> T {
        match self {
            Reactive::Variable(t) | Reactive::Static(t) => t,
            Reactive::Fn(f) => f(),
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
            Reactive::Variable(t) => f(t).v(),
            Reactive::Static(t) => f(t).s(),
            Reactive::Fn(t) => Reactive::Fn(Rc::new(move || f(t()))),
        }
    }
}

pub trait IntoReactive<T: 'static> {
    fn into_reactive(self) -> Reactive<T>;
}

impl<T, U> IntoReactive<T> for Reactive<U>
where
    T: 'static,
    U: 'static + Into<T>,
{
    fn into_reactive(self) -> Reactive<T> {
        self.cast()
    }
}

impl<T, F, U> IntoReactive<T> for F
where
    T: 'static,
    F: 'static + Fn() -> U,
    U: Into<T>,
{
    fn into_reactive(self) -> Reactive<T> {
        Reactive::Fn(Rc::new(move || self().into()))
    }
}

impl<T, U> IntoReactive<T> for Signal<U>
where
    T: 'static,
    U: 'static + Clone + Into<T>,
{
    fn into_reactive(self) -> Reactive<T> {
        ReadSignal::from(self).into_reactive()
    }
}

impl<T, U> IntoReactive<T> for ReadSignal<U>
where
    T: 'static,
    U: 'static + Clone + Into<T>,
{
    fn into_reactive(self) -> Reactive<T> {
        Reactive::Fn(Rc::new(move || self.get().into()))
    }
}

pub trait IntoReactiveValue<T>: Sized {
    fn into_variable(self) -> Reactive<T>;
    fn into_static(self) -> Reactive<T>;

    fn v(self) -> Reactive<T> {
        self.into_variable()
    }

    fn s(self) -> Reactive<T> {
        self.into_static()
    }
}

impl<T> IntoReactiveValue<T> for Reactive<T> {
    fn into_variable(self) -> Reactive<T> {
        if let Reactive::Static(t) = self {
            Reactive::Variable(t)
        } else {
            self
        }
    }

    fn into_static(self) -> Reactive<T> {
        if let Reactive::Variable(t) = self {
            Reactive::Static(t)
        } else {
            self
        }
    }
}

impl<T> IntoReactiveValue<T> for T {
    fn into_variable(self) -> Reactive<T> {
        Reactive::Variable(self)
    }

    fn into_static(self) -> Reactive<T> {
        Reactive::Static(self)
    }
}
