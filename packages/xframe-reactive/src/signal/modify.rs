use super::{Signal, SignalContext};
use std::{
    fmt,
    ops::{Deref, DerefMut},
};

impl<'a, T> Signal<'a, T> {
    pub fn modify(&self) -> Modify<'a, T> {
        Modify {
            value: self.inner.value.borrow_mut(),
            trigger: ModifyTrigger(&self.inner.context),
        }
    }
}

pub struct Modify<'a, T> {
    value: std::cell::RefMut<'a, T>,
    trigger: ModifyTrigger<'a>,
}

impl<T: fmt::Debug> fmt::Debug for Modify<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Modify").field(&self.value).finish()
    }
}

impl<'a, T> Modify<'a, T> {
    pub fn map<U, F>(this: Self, f: F) -> Modify<'a, U>
    where
        F: FnOnce(&mut T) -> &mut U,
    {
        let Modify { value, trigger } = this;
        Modify {
            value: std::cell::RefMut::map(value, f),
            trigger,
        }
    }

    pub fn drop_silent(this: Self) {
        let Modify { value, trigger } = this;
        drop(value);
        // Just a reference, it's cheap to forget it.
        std::mem::forget(trigger);
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

struct ModifyTrigger<'a>(&'a SignalContext<'a>);

impl Drop for ModifyTrigger<'_> {
    fn drop(&mut self) {
        self.0.trigger_subscribers();
    }
}
