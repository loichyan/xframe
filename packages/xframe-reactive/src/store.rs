use crate::Scope;
use std::ops::{Deref, DerefMut};

pub trait Store {
    type Input;

    fn new_in(cx: Scope, input: Self::Input) -> Self;
}

impl<'a> Scope<'a> {
    pub fn create_store<T: Store>(self, input: T::Input) -> &'a T {
        self.create_variable(T::new_in(self, input))
    }
}

#[derive(Debug)]
pub struct PlainStore<T>(T);

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
