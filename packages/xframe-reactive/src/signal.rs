mod modify;
pub use modify::Modify;

use crate::{
    arena::Owned,
    scope::{BoundedScopeShared, EffectRef, Scope},
};
use indexmap::IndexSet;
use std::{cell::RefCell, fmt};

pub struct Signal<'a, T> {
    inner: &'a SignalInner<'a, T>,
}

impl<T: fmt::Debug> fmt::Debug for Signal<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Signal")
            .field(&self.inner.value.borrow())
            .finish()
    }
}

impl<T> Clone for Signal<'_, T> {
    fn clone(&self) -> Self {
        Signal { inner: self.inner }
    }
}

impl<T> Copy for Signal<'_, T> {}

impl<'a, T> Signal<'a, T> {
    pub fn track(&self) {
        self.inner.context.track();
    }

    pub fn trigger_subscribers(&self) {
        self.inner.context.trigger_subscribers();
    }

    pub fn get(&self) -> Ref<'a, T> {
        self.track();
        self.get_untracked()
    }

    pub fn get_untracked(&self) -> Ref<'a, T> {
        Ref(self.inner.value.borrow())
    }

    pub fn set(&self, val: T) {
        self.set_slient(val);
        self.inner.context.trigger_subscribers();
    }

    pub fn set_slient(&self, val: T) {
        self.inner.value.replace(val);
    }

    pub fn update<F>(&self, f: F)
    where
        F: FnOnce(&mut T) -> T,
    {
        self.update_silent(f);
        self.inner.context.trigger_subscribers();
    }

    pub fn update_silent<F>(&self, f: F)
    where
        F: FnOnce(&mut T) -> T,
    {
        self.inner.value.replace_with(f);
    }
}

struct SignalInner<'a, T> {
    value: RefCell<T>,
    context: WrappedSignalContext<'a>,
}

struct WrappedSignalContext<'a> {
    owned: Owned<'static, SignalContext>,
    shared: BoundedScopeShared<'a>,
}

impl<'a> WrappedSignalContext<'a> {
    pub fn track(&self) {
        if let Some(eff) = self.shared.observer.get() {
            eff.with(|eff| eff.add_dependency(Owned::downgrade(&self.owned)));
        }
    }

    pub fn trigger_subscribers(&self) {
        let subscribers = self.owned.subscribers.replace_with(|subs| {
            IndexSet::with_capacity_and_hasher(subs.len(), Default::default())
        });
        // Effects attach to subscribers at the end of the effect scope,
        // an effect created inside another scope might send signals to its
        // outer scope, so we should ensure the inner effects re-execute
        // before outer ones to avoid potential double executions.
        for eff in subscribers {
            eff.with(|eff| eff.run());
        }
    }
}

#[derive(Default)]
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
}

pub struct Ref<'a, T>(std::cell::Ref<'a, T>);

impl<T: fmt::Debug> fmt::Debug for Ref<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<'a, T> Ref<'a, T> {
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
    pub fn create_signal<T: 'a>(self, t: T) -> Signal<'a, T> {
        let shared = self.shared();
        let context = {
            let owned = shared.singal_contexts.alloc(SignalContext::default());
            WrappedSignalContext { owned, shared }
        };
        let inner = self.create_variable(SignalInner {
            value: RefCell::new(t),
            context,
        });
        Signal { inner }
    }
}
