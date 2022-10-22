mod modify;
mod split;

pub use modify::Modify;
pub use split::{ReadSignal, WriteSignal};

use crate::{
    effect::RawEffect,
    scope::{Scope, ScopeShared},
    utils::ByAddress,
};
use indexmap::IndexSet;
use std::cell::RefCell;

#[derive(Debug)]
pub struct Signal<'a, T> {
    inner: &'a RawSignal<T>,
}

impl_clone_copy!(Signal['a, T]);

impl<'a, T> Signal<'a, T> {
    pub(crate) fn into_raw(self) -> &'a RawSignal<T> {
        self.inner
    }

    pub(crate) fn from_raw(t: &'a RawSignal<T>) -> Self {
        Signal { inner: t }
    }

    pub fn track(&self) {
        self.inner.context.track();
    }

    pub fn trigger_subscribers(&self) {
        self.inner.context.trigger_subscribers();
    }

    pub fn get(&self) -> Ref<'a, T> {
        self.inner.context.track();
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

#[derive(Debug)]
pub(crate) struct RawSignal<T> {
    value: RefCell<T>,
    context: SignalContext,
}

impl Drop for SignalContext {
    fn drop(&mut self) {
        // SAFETY: this will be dropped after disposing, it's safe to access it.
        let this: &'static SignalContext = unsafe { std::mem::transmute(&*self) };
        for eff in self.future_subscribers.get_mut().iter() {
            eff.0.remove_dependence(this);
        }
    }
}

type Subscribers = RefCell<IndexSet<ByAddress<'static, RawEffect<'static>>, ahash::RandomState>>;

#[derive(Debug)]
pub(crate) struct SignalContext {
    shared: ByAddress<'static, ScopeShared>,
    future_subscribers: Subscribers,
    subscribers: Subscribers,
}

impl SignalContext {
    pub fn track(&self) {
        if let Some(e) = self.shared.0.subscriber.get() {
            // SAFETY: An effect might captured a signal created inside the
            // child scope, we should notify the effect to remove the signal
            // to avoid access dangling pointer.
            self.future_subscribers.borrow_mut().insert(ByAddress(e));
            let this: &'static SignalContext = unsafe { std::mem::transmute(self) };
            e.add_dependence(this);
        }
    }

    pub fn subscribe(&self, effect: &'static RawEffect<'static>) {
        self.subscribers.borrow_mut().insert(ByAddress(effect));
    }

    pub fn unsubscribe(&self, effect: &'static RawEffect<'static>) {
        self.subscribers.borrow_mut().remove(&ByAddress(effect));
    }

    pub fn trigger_subscribers(&self) {
        self.future_subscribers.borrow_mut().clear();
        let subscribers = self.subscribers.replace_with(|subs| {
            IndexSet::with_capacity_and_hasher(subs.len(), Default::default())
        });
        // Effects attach to subscribers at the end of the effect scope,
        // an effect created inside another scope might send signals to its
        // outer scope, so we should ensure the inner effects re-execute
        // before outer ones to avoid potential double executions.
        for eff in subscribers {
            eff.0.run();
        }
    }
}

#[derive(Debug)]
pub struct Ref<'a, T>(std::cell::Ref<'a, T>);

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
        let inner = self.create_variable(RawSignal {
            value: RefCell::new(t),
            context: SignalContext {
                shared: ByAddress(self.shared()),
                future_subscribers: Default::default(),
                subscribers: Default::default(),
            },
        });
        Signal { inner }
    }
}
