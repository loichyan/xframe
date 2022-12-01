use super::Signal;
use crate::{
    shared::{SignalId, SHARED},
    variable::VarRefMut,
    CovariantLifetime,
};
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

impl<'a, T> Signal<'a, T> {
    pub fn modify(&self) -> SignalModify<'_, T> {
        SignalModify {
            value: self.get_mut(),
            trigger: ModifyTrigger {
                id: self.id,
                marker: PhantomData,
            },
        }
    }
}

pub struct SignalModify<'a, T> {
    value: VarRefMut<'a, T>,
    trigger: ModifyTrigger<'a>,
}

impl<'a, T> SignalModify<'a, T> {
    pub fn map<U>(this: Self, f: impl FnOnce(&mut T) -> &mut U) -> SignalModify<'a, U> {
        let SignalModify { value, trigger } = this;
        SignalModify {
            value: VarRefMut::map(value, f),
            trigger,
        }
    }
}

impl<'a, T> Deref for SignalModify<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<'a, T> DerefMut for SignalModify<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

struct ModifyTrigger<'a> {
    id: SignalId,
    marker: PhantomData<CovariantLifetime<'a>>,
}

impl Drop for ModifyTrigger<'_> {
    fn drop(&mut self) {
        SHARED.with(|shared| self.id.trigger(shared));
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
    fn map_modify() {
        create_root(|cx| {
            let state = cx.create_signal((1, 2, 3));
            let modify = state.modify();
            assert_eq!(&*modify, &(1, 2, 3));
            let field1 = SignalModify::map(modify, |(f1, _, _)| f1);
            assert_eq!(*field1, 1);
        });
    }
}
