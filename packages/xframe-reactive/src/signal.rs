mod modify;
pub use modify::Modify;

use crate::scope::{Cleanup, EffectId, Scope, Shared, SignalId};
use indexmap::IndexSet;
use std::{any::Any, cell::RefCell, fmt, marker::PhantomData};

pub struct Signal<'a, T> {
    inner: RawSignal<'a>,
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

impl<'a, T: 'static> Signal<'a, T> {
    fn value(&self) -> &'a RefCell<T> {
        self.inner
            .value()
            .downcast_ref()
            .unwrap_or_else(|| unreachable!())
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

pub(crate) type RawSignal<'a> = &'a dyn AnySignal;

pub(crate) trait AnySignal {
    fn this(&self) -> SignalId;
    fn shared(&self) -> &Shared;
    fn value(&self) -> &dyn Any;
    fn subscribers(&self) -> &RefCell<IndexSet<EffectId>>;
}

impl<'a> dyn 'a + AnySignal {
    pub fn subscribe(&self, id: EffectId) {
        self.subscribers().borrow_mut().insert(id);
    }

    pub fn unsubscribe(&self, id: EffectId) {
        self.subscribers().borrow_mut().remove(&id);
    }

    pub fn track(&self) {
        let shared = self.shared();
        let this = self.this();
        if let Some(id) = shared.observer.get() {
            shared
                .effects
                .borrow()
                .get(id)
                .map(|eff| eff.add_dependency(this));
        }
    }

    pub fn trigger_subscribers(&self) {
        let shared = self.shared();
        let subscribers = self.subscribers().take();
        // Effects attach to subscribers at the end of the effect scope,
        // an effect created inside another scope might send signals to its
        // outer scope, so we should ensure the inner effects re-execute
        // before outer ones to avoid potential double executions.
        for id in subscribers {
            shared.effects.borrow().get(id).map(|eff| eff.run());
        }
    }
}

pub(crate) struct AnySignalImpl<'a, T> {
    this: SignalId,
    shared: &'a Shared,
    value: RefCell<T>,
    subscribers: RefCell<IndexSet<EffectId>>,
}

impl<'a, T: 'static> AnySignal for AnySignalImpl<'a, T> {
    fn this(&self) -> SignalId {
        self.this
    }

    fn shared(&self) -> &Shared {
        self.shared
    }

    fn value(&self) -> &dyn Any {
        &self.value
    }

    fn subscribers(&self) -> &RefCell<IndexSet<EffectId>> {
        &self.subscribers
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
    fn create_signal_impl<T>(self, signal: &'a dyn AnySignal) -> Signal<'a, T> {
        self.push_cleanup(Cleanup::Signal(signal.this()));
        Signal {
            inner: signal,
            ty: PhantomData,
        }
    }

    pub fn create_signal<T: 'static>(self, t: T) -> Signal<'a, T> {
        let shared = self.shared();
        let mut signal = None;
        shared.signals.borrow_mut().insert_with_key(|id| {
            let any_impl = AnySignalImpl {
                this: id,
                shared,
                value: RefCell::new(t),
                subscribers: Default::default(),
            };
            let any = self.create_variable(any_impl) as &dyn AnySignal;
            signal = Some(any);
            unsafe { std::mem::transmute(any) }
        });
        self.create_signal_impl(signal.unwrap_or_else(|| unreachable!()))
    }
}
