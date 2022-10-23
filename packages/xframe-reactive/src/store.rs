use crate::{scope::Scope, signal::Signal};
use std::marker::PhantomData;

pub trait StoreBuilder<'a> {
    type Store;
    fn build_store(cx: Scope<'a>, this: Self) -> Self::Store;
}

pub struct CreateDefault<T>(pub PhantomData<T>);

impl<T> Default for CreateDefault<T> {
    fn default() -> Self {
        CreateDefault(PhantomData)
    }
}

impl<'a, T: Default> StoreBuilder<'a> for CreateDefault<T> {
    type Store = T;

    fn build_store(_cx: Scope<'a>, _this: Self) -> Self::Store {
        T::default()
    }
}

#[derive(Default)]
pub struct CreateSelf<T>(pub T);

impl<'a, T> StoreBuilder<'a> for CreateSelf<T> {
    type Store = T;

    fn build_store(_cx: Scope<'a>, this: Self) -> Self::Store {
        this.0
    }
}

#[derive(Default)]
pub struct CreateSignal<T>(pub T);

impl<'a, T: 'a> StoreBuilder<'a> for CreateSignal<T> {
    type Store = Signal<'a, T>;

    fn build_store(cx: Scope<'a>, this: Self) -> Self::Store {
        cx.create_signal(this.0)
    }
}

impl<'a> Scope<'a> {
    pub fn create_store<T>(self, t: T) -> &'a T::Store
    where
        T: StoreBuilder<'a>,
    {
        self.create_variable(T::build_store(self, t))
    }
}
