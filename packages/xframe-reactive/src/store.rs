use crate::{
    scope::Scope,
    signal::{RawSignal, Signal},
};
use std::marker::PhantomData;

pub trait Store<'a> {
    type Source;
    type Output;

    fn create_source(cx: Scope<'a>, this: Self) -> Self::Source;
    fn map_source(source: &'a Self::Source) -> Self::Output;
}

pub struct DefaultStore<T>(pub PhantomData<T>);

impl<T> Default for DefaultStore<T> {
    fn default() -> Self {
        DefaultStore(PhantomData)
    }
}

impl<'a, T: 'a + Default> Store<'a> for DefaultStore<T> {
    type Source = T;
    type Output = &'a T;

    fn create_source(_cx: Scope<'a>, _this: Self) -> Self::Source {
        T::default()
    }

    fn map_source(source: &'a Self::Source) -> Self::Output {
        &source
    }
}

pub struct PlainStore<T>(pub T);

impl<'a, T: 'a> Store<'a> for PlainStore<T> {
    type Source = T;
    type Output = &'a T;

    fn create_source(_cx: Scope<'a>, this: Self) -> Self::Source {
        this.0
    }

    fn map_source(source: &'a Self::Source) -> Self::Output {
        &source
    }
}

pub struct ReactiveStore<T>(pub T);
pub struct ReactiveStoreSource<'a, T>(RawSignal<'a, T>);

impl<'a, T: 'a> Store<'a> for ReactiveStore<T> {
    type Source = ReactiveStoreSource<'a, T>;
    type Output = Signal<'a, T>;

    fn create_source(cx: Scope<'a>, this: Self) -> Self::Source {
        ReactiveStoreSource(RawSignal::new(cx, this.0))
    }

    fn map_source(source: &'a Self::Source) -> Self::Output {
        Signal::from_raw(&source.0)
    }
}

impl<'a> Scope<'a> {
    pub fn create_store<T>(self, t: T) -> T::Output
    where
        T: Store<'a>,
        T::Source: 'a,
    {
        let source = self.create_variable(T::create_source(self, t));
        T::map_source(source)
    }
}
