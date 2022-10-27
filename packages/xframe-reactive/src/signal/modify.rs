use super::{RawSignal, Signal};
use std::{
    fmt,
    ops::{Deref, DerefMut},
};

impl<'a, T> Signal<'a, T> {
    pub fn modify(&self) -> Modify<'a, T> {
        Modify {
            value: self.value().borrow_mut(),
            trigger: ModifyTrigger(self.inner),
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
    pub fn map<U>(this: Self, f: impl FnOnce(&mut T) -> &mut U) -> Modify<'a, U> {
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

struct ModifyTrigger<'a>(&'a RawSignal<'a>);

impl Drop for ModifyTrigger<'_> {
    fn drop(&mut self) {
        self.0.trigger_subscribers();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Scope;

    #[test]
    fn signal_modify() {
        Scope::create_root(|cx| {
            let state = cx.create_signal(String::from("Hello, "));
            let counter = cx.create_signal(0);
            cx.create_effect(move |_| {
                state.track();
                counter.update(|x| *x + 1);
            });
            assert_eq!(*counter.get(), 1);
            *state.modify() += "xFrame!";
            assert_eq!(*state.get(), "Hello, xFrame!");
            assert_eq!(*counter.get(), 2);
        });
    }

    #[test]
    fn signal_modify_silent() {
        Scope::create_root(|cx| {
            let state = cx.create_signal(String::from("Hello, "));
            let counter = cx.create_signal(0);
            cx.create_effect(move |_| {
                state.track();
                counter.update(|x| *x + 1);
            });
            assert_eq!(*counter.get(), 1);
            let mut modify = state.modify();
            *modify += "xFrame!";
            Modify::drop_silent(modify);
            assert_eq!(*state.get(), "Hello, xFrame!");
            assert_eq!(*counter.get(), 1);
        });
    }
}
