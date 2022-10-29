mod modify;
pub use modify::SignalModify;

use crate::{
    scope::{Cleanup, OwnedScope},
    shared::{EffectRef, Shared, SignalContextRef},
};
use indexmap::IndexSet;
use std::{cell::RefCell, fmt, ops::Deref};

pub type Signal<'a, T> = &'a OwnedSignal<'a, T>;
pub type ReadSignal<'a, T> = &'a OwnedReadSignal<'a, T>;

pub struct OwnedReadSignal<'a, T> {
    value: RefCell<T>,
    context: &'a SignalContext,
}

impl<T: fmt::Debug> fmt::Debug for OwnedReadSignal<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Signal").field(&self.value.borrow()).finish()
    }
}

impl<'a, T> OwnedReadSignal<'a, T> {
    pub fn track(&self) {
        self.context.track();
    }

    pub fn trigger_subscribers(&self) {
        self.context.trigger_subscribers();
    }

    pub fn get(&self) -> SignalRef<'_, T> {
        self.track();
        self.get_untracked()
    }

    pub fn get_untracked(&self) -> SignalRef<'_, T> {
        SignalRef(self.value.borrow())
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
    pub fn set(&self, val: T) {
        self.set_slient(val);
        self.trigger_subscribers();
    }

    pub fn set_slient(&self, val: T) {
        self.value.replace(val);
    }

    pub fn update(&self, f: impl FnOnce(&mut T) -> T) {
        self.update_silent(f);
        self.trigger_subscribers();
    }

    pub fn update_silent(&self, f: impl FnOnce(&mut T) -> T) {
        self.value.replace_with(f);
    }
}

pub(crate) struct SignalContext {
    this: SignalContextRef,
    shared: &'static Shared,
    subscribers: RefCell<IndexSet<EffectRef>>,
}

impl SignalContext {
    pub fn track(&self) {
        if let Some(eff) = self.shared.observer.get() {
            eff.with(|eff| eff.add_dependency(self.this));
        }
    }

    pub fn subscribe(&self, effect: EffectRef) {
        self.subscribers.borrow_mut().insert(effect);
    }

    pub fn unsubscribe(&self, effect: EffectRef) {
        self.subscribers.borrow_mut().remove(&effect);
    }

    pub fn trigger_subscribers(&self) {
        let subscribers = self.subscribers.take();
        // Effects attach to subscribers at the end of the effect scope,
        // an effect created inside another scope might send signals to its
        // outer scope, so we should ensure the inner effects re-execute
        // before outer ones to avoid potential double executions.
        for eff in subscribers {
            eff.with(|eff| eff.run());
        }
    }
}

pub struct SignalRef<'a, T: ?Sized>(std::cell::Ref<'a, T>);

impl<T: fmt::Debug> fmt::Debug for SignalRef<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<'a, T: ?Sized> SignalRef<'a, T> {
    pub fn map<U>(orig: Self, f: impl FnOnce(&T) -> &U) -> SignalRef<'a, U> {
        SignalRef(std::cell::Ref::map(orig.0, f))
    }
}

impl<'a, T> Deref for SignalRef<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl<'a> OwnedScope<'a> {
    fn create_signal_context(&self) -> &'a SignalContext {
        let shared = self.shared();
        let ctx = shared
            .signal_contexts
            .alloc_with_weak(|this| SignalContext {
                this,
                shared,
                subscribers: Default::default(),
            });
        self.push_cleanup(Cleanup::SignalContext(ctx));
        // SAFETY: the signal will be freed and no longer accessible for weak
        // references once current scope is disposed.
        unsafe { ctx.leak_ref() }
    }

    pub fn create_owned_read_signal<T>(&'a self, t: T) -> OwnedReadSignal<'a, T> {
        OwnedReadSignal {
            value: RefCell::new(t),
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
                let state = unsafe { cx.create_variable_unchecked(owned) };
                let double = cx.create_memo(|| *state.get() * 2);
                assert_eq!(*state.get(), 1);
                assert_eq!(*double.get(), 2);
                state.set(2);
                assert_eq!(*state.get(), 2);
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
                unsafe { cx.create_variable_unchecked(owned) };
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
            let context = owned.context.this;
            cx.create_child(|cx| {
                unsafe { cx.create_variable_unchecked(owned) };
            });
            assert!(context.can_upgrade());
        });
    }
}
