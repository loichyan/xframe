use super::{Ref, Signal};
use crate::Modify;

impl<'a, T> Signal<'a, T> {
    pub fn split(&self) -> (ReadSignal<'a, T>, WriteSignal<'a, T>) {
        let inner = *self;
        (ReadSignal { inner }, WriteSignal { inner })
    }
}

#[derive(Debug)]
pub struct ReadSignal<'a, T> {
    inner: Signal<'a, T>,
}

impl_clone_copy!(ReadSignal['a, T]);

impl<'a, T> ReadSignal<'a, T> {
    pub fn get_untracked(&self) -> Ref<T> {
        self.inner.get_untracked()
    }

    pub fn get(&self) -> Ref<T> {
        self.inner.get()
    }
}

#[derive(Debug)]
pub struct WriteSignal<'a, T> {
    inner: Signal<'a, T>,
}

impl_clone_copy!(WriteSignal['a, T]);

impl<'a, T> WriteSignal<'a, T> {
    pub fn set_slient(&self, val: T) {
        self.inner.set_slient(val);
    }

    pub fn set(&self, val: T) {
        self.inner.set(val);
    }

    pub fn update_slient<F>(&self, f: F)
    where
        F: FnOnce(&mut T) -> T,
    {
        self.inner.update_silent(f);
    }

    pub fn update<F>(&self, f: F)
    where
        F: FnOnce(&mut T) -> T,
    {
        self.inner.update(f);
    }

    pub fn modify(&self) -> Modify<'a, T> {
        self.inner.modify()
    }
}
