mod modify;
pub use modify::SignalModify;

use crate::{
    scope::{Cleanup, OwnedScope},
    shared::{EffectRef, Shared, SignalContextRef},
    variable::VarRef,
    OwnedVariable, VarRefMut,
};
use indexmap::IndexSet;
use std::{cell::RefCell, fmt, ops::Deref};

pub type Signal<'a, T> = &'a OwnedSignal<'a, T>;
pub type ReadSignal<'a, T> = &'a OwnedReadSignal<'a, T>;

pub struct OwnedReadSignal<'a, T> {
    value: OwnedVariable<'a, T>,
    shared: &'static Shared,
    context: SignalContextRef,
}

impl<T: fmt::Debug> fmt::Debug for OwnedReadSignal<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Signal").field(&*self.value.get()).finish()
    }
}

impl<'a, T> OwnedReadSignal<'a, T> {
    pub fn track(&self) {
        if let Some(eff) = self.shared.observer.get() {
            eff.try_get().map(|eff| eff.add_dependency(self.context));
        }
    }

    pub fn trigger_subscribers(&self) {
        self.context.get().trigger_subscribers(self.shared);
    }

    pub fn get(&self) -> VarRef<'_, T> {
        self.track();
        self.get_untracked()
    }

    pub fn get_untracked(&self) -> VarRef<'_, T> {
        self.value.get()
    }
}

pub struct OwnedSignal<'a, T>(OwnedReadSignal<'a, T>);

impl<'a, T> Deref for OwnedSignal<'a, T> {
    type Target = OwnedReadSignal<'a, T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: fmt::Debug> fmt::Debug for OwnedSignal<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<'a, T> OwnedSignal<'a, T> {
    pub fn get_mut(&self) -> VarRefMut<'_, T> {
        self.value.get_mut()
    }

    pub fn set(&self, val: T) {
        self.set_slient(val);
        self.trigger_subscribers();
    }

    pub fn set_slient(&self, val: T) {
        *self.value.get_mut() = val;
    }

    pub fn update(&self, f: impl FnOnce(&mut T) -> T) {
        self.update_silent(f);
        self.trigger_subscribers();
    }

    pub fn update_silent(&self, f: impl FnOnce(&mut T) -> T) {
        let val = &mut *self.value.get_mut();
        *val = f(val);
    }
}

pub(crate) struct SignalContext {
    subscribers: RefCell<IndexSet<EffectRef>>,
}

impl SignalContext {
    pub fn subscribe(&self, effect: EffectRef) {
        self.subscribers.borrow_mut().insert(effect);
    }

    pub fn unsubscribe(&self, effect: EffectRef) {
        self.subscribers.borrow_mut().remove(&effect);
    }

    pub fn trigger_subscribers(&self, shared: &Shared) {
        let subscribers = self
            .subscribers
            .replace_with(|sub| IndexSet::with_capacity_and_hasher(sub.len(), Default::default()));
        // Effects attach to subscribers at the end of the effect scope,
        // an effect created inside another scope might send signals to its
        // outer scope, so we should ensure the inner effects re-execute
        // before outer ones to avoid potential double executions.
        for eff in subscribers {
            eff.try_get().map(|raw| raw.run(eff, shared));
        }
    }
}

impl<'a> OwnedScope<'a> {
    fn create_signal_context(&'a self) -> SignalContextRef {
        let shared = self.shared();
        let ctx = shared.signal_contexts.alloc(SignalContext {
            subscribers: Default::default(),
        });
        self.push_cleanup(Cleanup::SignalContext(ctx));
        ctx
    }

    pub fn create_owned_read_signal<T>(&'a self, t: T) -> OwnedReadSignal<'a, T> {
        OwnedReadSignal {
            shared: self.shared(),
            value: self.create_owned_variable(t),
            context: self.create_signal_context(),
        }
    }

    pub fn create_owned_signal<T>(&'a self, t: T) -> OwnedSignal<'a, T> {
        OwnedSignal(self.create_owned_read_signal(t))
    }

    pub fn create_read_signal<T>(&'a self, t: T) -> ReadSignal<'a, T> {
        // SAFETY: The `ReadSignal` itself will not access its underlying value.
        unsafe { self.create_variable_unchecked(self.create_owned_read_signal(t)) }
    }

    pub fn create_signal<T>(&'a self, t: T) -> Signal<'a, T> {
        // SAFETY: Same as `create_read_signal`.
        unsafe { self.create_variable_unchecked(self.create_owned_signal(t)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_root;
    use std::cell::Cell;

    #[test]
    fn reactive_signal() {
        create_root(|cx| {
            let state = cx.create_signal(0);
            assert_eq!(*state.get(), 0);
            state.set(1);
            assert_eq!(*state.get(), 1);
        });
    }

    #[test]
    fn signal_composition() {
        create_root(|cx| {
            let state = cx.create_signal(1);
            let double = || *state.get() * 2;
            assert_eq!(double(), 2);
            state.set(2);
            assert_eq!(double(), 4);
        });
    }

    #[test]
    fn signal_set_slient() {
        create_root(|cx| {
            let state = cx.create_signal(1);
            let double = cx.create_memo(|| *state.get() * 2);
            assert_eq!(*double.get(), 2);
            state.set_slient(2);
            assert_eq!(*double.get(), 2);
        });
    }

    #[test]
    fn signal_of_signal() {
        create_root(|cx| {
            let state = cx.create_signal(1);
            let state2 = cx.create_signal(state);
            let double = cx.create_memo(|| *state2.get().get() * 2);
            assert_eq!(*state2.get().get(), 1);
            state.set(2);
            assert_eq!(*double.get(), 4);
        });
    }

    #[test]
    fn owned_signal_in_child_scope() {
        create_root(|cx| {
            let owned = cx.create_owned_signal(1);
            cx.create_child(|cx| {
                let state = cx.create_variable(owned);
                let double = cx.create_memo(|| *state.get().get() * 2);
                assert_eq!(*state.get().get(), 1);
                assert_eq!(*double.get(), 2);
                state.get().set(2);
                assert_eq!(*state.get().get(), 2);
                assert_eq!(*double.get(), 4);
            });
        });
    }

    #[test]
    fn dispose_owned_signal_in_child_scope() {
        thread_local! {
            static COUNTER: Cell<i32> = Cell::new(0);
        }

        struct DropAndInc;
        impl Drop for DropAndInc {
            fn drop(&mut self) {
                COUNTER.with(|x| x.set(x.get() + 1));
            }
        }

        struct DropAndAssert(i32);
        impl Drop for DropAndAssert {
            fn drop(&mut self) {
                assert_eq!(COUNTER.with(Cell::get), self.0);
            }
        }

        create_root(|cx| {
            let owned = cx.create_owned_signal(DropAndInc);
            cx.create_child(|cx| {
                cx.create_variable(DropAndAssert(1));
                cx.create_variable(owned);
                cx.create_variable(DropAndAssert(0));
            });
            cx.create_owned_signal(DropAndInc);
        });
        drop(DropAndAssert(2));
    }

    #[test]
    fn can_read_signal_context_after_dispose_in_child() {
        create_root(|cx| {
            let owned = cx.create_owned_signal(0);
            let context = owned.context;
            cx.create_child(|cx| {
                cx.create_variable(owned);
            });
            assert!(context.can_upgrade());
        });
    }

    #[test]
    #[should_panic = "get a disposed variable"]
    fn cannot_read_a_disposed_signal_value() {
        struct DropAndRead<'a>(Option<Signal<'a, String>>);
        impl Drop for DropAndRead<'_> {
            fn drop(&mut self) {
                self.0.unwrap().get();
            }
        }

        create_root(|cx| {
            let var = cx.create_variable(DropAndRead(None));
            let signal = cx.create_signal(String::from("Hello, xFrame!"));
            var.get_mut().0 = Some(signal);
        });
    }

    #[test]
    #[should_panic = "get a disposed slot"]
    fn cannot_read_a_disposed_signal_context() {
        struct DropAndRead<'a>(Option<Signal<'a, String>>);
        impl Drop for DropAndRead<'_> {
            fn drop(&mut self) {
                self.0.unwrap().trigger_subscribers();
            }
        }

        create_root(|cx| {
            let var = cx.create_variable(DropAndRead(None));
            let signal = cx.create_signal(String::from("Hello, xFrame!"));
            var.get_mut().0 = Some(signal);
        });
    }
}
