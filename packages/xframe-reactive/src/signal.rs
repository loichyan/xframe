mod modify;
pub use modify::Modify;

use crate::scope::{Cleanup, EffectRef, Scope, Shared, SignalRef};
use indexmap::IndexSet;
use std::{any::Any, cell::RefCell, fmt, marker::PhantomData};

pub struct Signal<'a, T> {
    inner: SignalRef,
    ty: PhantomData<&'a T>,
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

impl<'a, T: 'static> Signal<'a, T> {
    fn with<U>(&self, f: impl FnOnce(&RawSignal<'static>) -> U) -> U {
        self.inner.with(f).unwrap_or_else(|| unreachable!())
    }

    fn value(&self) -> &'a RefCell<T> {
        self.with(|raw| raw.value.downcast_ref().unwrap_or_else(|| unreachable!()))
    }

    pub fn track(&self) {
        self.inner.with(|raw| {
            if let Some(eff) = raw.shared.observer.get() {
                eff.with(|raw| raw.add_dependency(self.inner));
            }
        });
    }

    pub fn trigger_subscribers(&self) {
        self.inner.with(|raw| {
            raw.trigger_subscribers();
        });
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
    shared: &'static Shared,
    value: &'a dyn Any,
    subscribers: RefCell<IndexSet<EffectRef>>,
}

impl<'a> RawSignal<'a> {
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
            eff.with(|raw| raw.run(eff));
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
    fn create_signal_impl<T>(self, value: &'a dyn Any) -> Signal<'a, T> {
        let shared = self.shared();
        let raw = RawSignal {
            shared,
            value,
            subscribers: Default::default(),
        };
        // SAFETY: the signal will be freed and no longer accessible for weak
        // references once current scope is disposed.
        let raw = unsafe { std::mem::transmute(raw) };
        let inner = shared.signals.alloc(raw);
        self.push_cleanup(Cleanup::Signal(inner));
        Signal {
            inner,
            ty: PhantomData,
        }
    }

    pub fn create_signal<T: 'static>(self, t: T) -> Signal<'a, T> {
        let value = self.create_variable(RefCell::new(t));
        self.create_signal_impl(value)
    }
}
