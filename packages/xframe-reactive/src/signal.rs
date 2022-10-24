mod modify;
pub use modify::Modify;

use crate::{
    effect::RawEffect,
    scope::{Scope, ScopeShared},
    utils::ByAddress,
};
use ahash::AHashSet;
use indexmap::IndexSet;
use once_cell::sync::OnceCell;
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
    context: SignalContext<'a>,
}

pub(crate) struct SignalContext<'a> {
    this: OnceCell<&'static SignalContext<'static>>,
    shared: &'a ScopeShared,
    future_subscribers: RefCell<AHashSet<ByAddress<'static, RawEffect<'static>>>>,
    subscribers: RefCell<IndexSet<ByAddress<'static, RawEffect<'static>>, ahash::RandomState>>,
}

impl Drop for SignalContext<'_> {
    fn drop(&mut self) {
        // SAFETY: this will be dropped after disposing, it's safe to access it.
        let this = self.this();
        for eff in self.future_subscribers.get_mut().iter() {
            eff.0.remove_dependence(this);
        }
    }
}

impl<'a> SignalContext<'a> {
    fn this(&self) -> &'static SignalContext<'static> {
        let this = *self.this.get().unwrap_or_else(|| unreachable!());
        debug_assert_eq!(
            this as *const SignalContext as *const (),
            self as *const SignalContext as *const ()
        );
        this
    }

    pub fn track(&self) {
        if let Some(e) = self.shared.observer.get() {
            // SAFETY: An effect might captured a signal created inside the
            // child scope, we should notify the effect to remove the signal
            // to avoid access dangling pointer.
            let this = self.this();
            // The signal might be used as a dependence in the future, we should
            // record this effect, and notify it of unsubscribing when the signal
            // is disposed.
            self.future_subscribers.borrow_mut().insert(ByAddress(e));
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
        let inner = self.create_variable(SignalInner {
            value: RefCell::new(t),
            context: SignalContext {
                this: OnceCell::default(),
                shared: self.shared(),
                future_subscribers: Default::default(),
                subscribers: Default::default(),
            },
        });
        // SAFETY: crated variable has a stable address and lives as long
        // as the scope.
        let ptr = unsafe { std::mem::transmute(&inner.context) };
        inner
            .context
            .this
            .set(ptr)
            .unwrap_or_else(|_| unreachable!());
        Signal { inner }
    }
}
