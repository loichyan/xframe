use super::{OwnedSignal, SignalContext};
use std::{
    cell::RefMut,
    fmt,
    ops::{Deref, DerefMut},
};

impl<'a, T> OwnedSignal<'a, T> {
    pub fn modify(&self) -> SignalModify<'_, T> {
        SignalModify {
            value: self.value.borrow_mut(),
            trigger: ModifyTrigger(self.context),
        }
    }
}

pub struct SignalModify<'a, T> {
    value: std::cell::RefMut<'a, T>,
    trigger: ModifyTrigger<'a>,
}

impl<T: fmt::Debug> fmt::Debug for SignalModify<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Modify").field(&self.value).finish()
    }
}

impl<'a, T> SignalModify<'a, T> {
    pub fn map<U>(this: Self, f: impl FnOnce(&mut T) -> &mut U) -> SignalModify<'a, U> {
        let SignalModify { value, trigger } = this;
        SignalModify {
            value: RefMut::map(value, f),
            trigger,
        }
    }

    pub fn drop_silent(this: Self) {
        let SignalModify { value, trigger } = this;
        drop(value);
        // Just a reference, it's cheap to forget it.
        std::mem::forget(trigger);
    }
}

impl<'a, T> Deref for SignalModify<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &*self.value
    }
}

impl<'a, T> DerefMut for SignalModify<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.value
    }
}

struct ModifyTrigger<'a>(&'a SignalContext);

impl Drop for ModifyTrigger<'_> {
    fn drop(&mut self) {
        self.0.trigger_subscribers();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_root;

    #[test]
    fn signal_modify() {
        create_root(|cx| {
            let state = cx.create_signal(String::from("Hello, "));
            let counter = cx.create_signal(0);
            cx.create_effect(|_| {
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
        create_root(|cx| {
            let state = cx.create_signal(String::from("Hello, "));
            let counter = cx.create_signal(0);
            cx.create_effect(|_| {
                state.track();
                counter.update(|x| *x + 1);
            });
            assert_eq!(*counter.get(), 1);
            let mut modify = state.modify();
            *modify += "xFrame!";
            SignalModify::drop_silent(modify);
            assert_eq!(*state.get(), "Hello, xFrame!");
            assert_eq!(*counter.get(), 1);
        });
    }
}
