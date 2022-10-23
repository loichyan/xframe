mod modify;

use crate::{
    effect::RawEffect,
    scope::{Scope, ScopeShared},
    store::Store,
    utils::ByAddress,
};
use ahash::AHashSet;
use indexmap::IndexSet;
use std::{cell::RefCell, ops::Deref};

pub use modify::Modify;

#[derive(Debug)]
pub struct ReadSignal<T> {
    value: RefCell<T>,
    context: SignalContext,
}

impl<T> Store for ReadSignal<T> {
    type Input = T;

    fn new_in(cx: Scope, input: Self::Input) -> Self {
        ReadSignal {
            value: RefCell::new(input),
            context: SignalContext {
                shared: ByAddress(cx.shared()),
                future_subscribers: Default::default(),
                subscribers: Default::default(),
            },
        }
    }
}

impl<T> ReadSignal<T> {
    pub fn track(&self) {
        self.context.track();
    }

    pub fn trigger_subscribers(&self) {
        self.context.trigger_subscribers();
    }

    pub fn get(&self) -> Ref<T> {
        self.context.track();
        self.get_untracked()
    }

    pub fn get_untracked(&self) -> Ref<T> {
        Ref(self.value.borrow())
    }
}

#[derive(Debug)]
pub struct Signal<T> {
    inner: ReadSignal<T>,
}

impl<T> Store for Signal<T> {
    type Input = T;

    fn new_in(cx: Scope, input: Self::Input) -> Self {
        Self {
            inner: ReadSignal::new_in(cx, input),
        }
    }
}

impl<T> Signal<T> {
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

impl<T> std::ops::Deref for Signal<T> {
    type Target = ReadSignal<T>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[derive(Debug)]
pub(crate) struct SignalContext {
    shared: ByAddress<'static, ScopeShared>,
    future_subscribers: RefCell<AHashSet<ByAddress<'static, RawEffect<'static>>>>,
    subscribers: RefCell<IndexSet<ByAddress<'static, RawEffect<'static>>, ahash::RandomState>>,
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

impl SignalContext {
    pub fn track(&self) {
        if let Some(e) = self.shared.0.observer.get() {
            // SAFETY: An effect might captured a signal created inside the
            // child scope, we should notify the effect to remove the signal
            // to avoid access dangling pointer.
            let this: &'static SignalContext = unsafe { std::mem::transmute(self) };
            // The signal might be used as a dependence in the future, we should
            // record this effect, and notify it of unsubscribing when the signal
            // is disposed.
            self.future_subscribers.borrow_mut().insert(ByAddress(e.0));
            e.0.add_dependence(this);
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
        let subscribers = self.subscribers.take();
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

impl<'a, T> Deref for Ref<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl<'a> Scope<'a> {
    pub fn create_signal<T: 'a>(self, t: T) -> &'a Signal<T> {
        self.create_variable(Signal::new_in(self, t))
    }
}
