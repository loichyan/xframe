use crate::Scope;
use std::{
    fmt,
    ops::{Deref, DerefMut},
};

pub trait Store {
    type Input;

    fn new_in(cx: Scope, input: Self::Input) -> Self;
}

impl<'a> Scope<'a> {
    pub fn create_store<T: Store>(self, input: T::Input) -> &'a T {
        self.create_variable(T::new_in(self, input))
    }
}

pub struct PlainStore<T>(T);

impl<T: fmt::Debug> fmt::Debug for PlainStore<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<T> Store for PlainStore<T> {
    type Input = T;

    fn new_in(_: Scope, input: Self::Input) -> Self {
        PlainStore(input)
    }
}

impl<T> Deref for PlainStore<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for PlainStore<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
