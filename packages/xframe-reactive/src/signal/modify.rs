use super::{Signal, SignalContext};
use std::{
    mem::ManuallyDrop,
    ops::{Deref, DerefMut},
};

impl<T> Signal<T> {
    pub fn modify(&self) -> Modify<T> {
        Modify {
            value: ManuallyDrop::new(self.inner.value.borrow_mut()),
            context: &self.inner.context,
        }
    }
}

#[derive(Debug)]
pub struct Modify<'a, T> {
    value: ManuallyDrop<std::cell::RefMut<'a, T>>,
    context: &'a SignalContext,
}

impl<'a, T> Modify<'a, T> {
    fn take_value(mut this: Self) -> std::cell::RefMut<'a, T> {
        // SAFETY: this is forgotten immediately and value will never be accessed.
        let value = unsafe { ManuallyDrop::take(&mut this.value) };
        std::mem::forget(this);
        value
    }

    pub fn map<U, F>(this: Self, f: F) -> Modify<'a, U>
    where
        F: FnOnce(&mut T) -> &mut U,
    {
        let context = this.context;
        let value = Modify::take_value(this);
        Modify {
            value: ManuallyDrop::new(std::cell::RefMut::map(value, f)),
            context,
        }
    }

    pub fn drop_silent(this: Self) {
        drop(Modify::take_value(this));
    }
}

impl<'a, T> Deref for Modify<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &*self.value
    }
}

impl<'a, T> DerefMut for Modify<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.value
    }
}

impl<'a, T> Drop for Modify<'a, T> {
    fn drop(&mut self) {
        // SAFETY: this variable is dropped and will never be accessed.
        let value = unsafe { ManuallyDrop::take(&mut self.value) };
        drop(value);
        self.context.trigger_subscribers();
    }
}
