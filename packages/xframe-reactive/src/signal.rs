mod modify;
pub use modify::Modify;

use crate::{
    scope::{Cleanup, EffectRef, Scope, Shared, SignalRef},
    Empty,
};
use indexmap::IndexSet;
use std::{cell::RefCell, fmt, marker::PhantomData};

pub struct Signal<'a, T> {
    inner: &'a RawSignal<'a>,
    ty: PhantomData<T>,
}

impl<T: fmt::Debug> fmt::Debug for Signal<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Signal")
            // .field(&self.inner.value.borrow())
            .finish()
    }
}

impl<T> Clone for Signal<'_, T> {
    fn clone(&self) -> Self {
        Signal { ..*self }
    }
}

impl<T> Copy for Signal<'_, T> {}

impl<'a, T> Signal<'a, T> {
    fn value(&self) -> &'a RefCell<T> {
        // SAFETY: the type is guaranteed by self.ty
        unsafe { &*(self.inner.value as *const dyn Empty as *const RefCell<T>) }
    }

    pub fn track(&self) {
        self.inner.track();
    }

    pub fn trigger_subscribers(&self) {
        self.inner.trigger_subscribers();
    }

    pub fn get(&self) -> Ref<'a, T> {
        self.track();
        self.get_untracked()
    }

    pub fn get_untracked(&self) -> Ref<'a, T> {
        Ref(self.value().borrow())
    }

    pub fn set(&self, val: T) {
        self.set_slient(val);
        self.trigger_subscribers();
    }

    pub fn set_slient(&self, val: T) {
        self.value().replace(val);
    }

    pub fn update<F>(&self, f: F)
    where
        F: FnOnce(&mut T) -> T,
    {
        self.update_silent(f);
        self.trigger_subscribers();
    }

    pub fn update_silent<F>(&self, f: F)
    where
        F: FnOnce(&mut T) -> T,
    {
        self.value().replace_with(f);
    }
}

pub(crate) struct RawSignal<'a> {
    this: SignalRef,
    shared: &'static Shared,
    value: &'a (dyn 'a + Empty),
    subscribers: RefCell<IndexSet<EffectRef>>,
}

impl<'a> RawSignal<'a> {
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

pub struct Ref<'a, T: ?Sized>(std::cell::Ref<'a, T>);

impl<T: fmt::Debug> fmt::Debug for Ref<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<'a, T: ?Sized> Ref<'a, T> {
    pub fn map<U, F>(orig: Self, f: F) -> Ref<'a, U>
    where
        F: FnOnce(&T) -> &U,
    {
        Ref(std::cell::Ref::map(orig.0, f))
    }
}

impl<'a, T> std::ops::Deref for Ref<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl<'a> Scope<'a> {
    fn create_signal_impl<T>(self, value: &'a (dyn 'a + Empty)) -> Signal<'a, T> {
        let shared = self.shared();
        let raw = shared.signals.alloc_with_weak(|this| {
            let raw = RawSignal {
                this,
                shared,
                value,
                subscribers: Default::default(),
            };
            // SAFETY: the signal will be freed and no longer accessible for weak
            // references once current scope is disposed.
            unsafe { std::mem::transmute(raw) }
        });
        let inner = unsafe { raw.leak_ref() };
        self.push_cleanup(Cleanup::Signal(raw));
        Signal {
            inner,
            ty: PhantomData,
        }
    }

    pub fn create_signal<T: 'a>(self, t: T) -> Signal<'a, T> {
        let value = self.create_variable(RefCell::new(t));
        self.create_signal_impl(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reactive_signal() {
        Scope::create_root(|cx| {
            let state = cx.create_signal(0);
            assert_eq!(*state.get(), 0);
            state.set(1);
            assert_eq!(*state.get(), 1);
        });
    }

    #[test]
    fn signal_composition() {
        Scope::create_root(|cx| {
            let state = cx.create_signal(1);
            let double = || *state.get() * 2;
            assert_eq!(double(), 2);
            state.set(2);
            assert_eq!(double(), 4);
        });
    }

    #[test]
    fn signal_set_slient() {
        Scope::create_root(|cx| {
            let state = cx.create_signal(1);
            let double = cx.create_memo(move || *state.get() * 2);
            assert_eq!(*double.get(), 2);
            state.set_slient(2);
            assert_eq!(*double.get(), 2);
        });
    }

    #[test]
    fn signal_of_signal() {
        Scope::create_root(|cx| {
            let state = cx.create_signal(1);
            let state2 = cx.create_signal(state);
            let double = cx.create_memo(move || *state2.get().get() * 2);
            assert_eq!(*state2.get().get(), 1);
            state.set(2);
            assert_eq!(*double.get(), 4);
        });
    }
}
