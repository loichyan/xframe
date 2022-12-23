use crate::{
    runtime::{EffectId, SignalId, RT},
    scope::{Cleanup, Scope},
    ThreadLocal,
};
use indexmap::IndexSet;
use smallvec::SmallVec;
use std::{any::Any, cell::RefCell, fmt, marker::PhantomData, ops::Deref, rc::Rc};

const INITIAL_SUBCRIBER_SLOTS: usize = 4;

pub(crate) type RawSignal = Rc<RefCell<dyn Any>>;

pub struct ReadSignal<T> {
    pub(crate) id: SignalId,
    marker: PhantomData<(T, ThreadLocal)>,
}

impl<T: 'static + fmt::Debug> fmt::Debug for ReadSignal<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.read(|v| f.debug_tuple("Signal").field(v).finish())
    }
}

impl<T: 'static + fmt::Display> fmt::Display for ReadSignal<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.read(|v| v.fmt(f))
    }
}

impl<T> Clone for ReadSignal<T> {
    fn clone(&self) -> Self {
        Self { ..*self }
    }
}

impl<T> Copy for ReadSignal<T> {}

impl<T> From<Signal<T>> for ReadSignal<T> {
    fn from(t: Signal<T>) -> Self {
        t.0
    }
}

pub struct Signal<T>(ReadSignal<T>);

impl<T: 'static + fmt::Debug> fmt::Debug for Signal<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.deref().fmt(f)
    }
}

impl<T: 'static + fmt::Display> fmt::Display for Signal<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.deref().fmt(f)
    }
}

impl<T> Deref for Signal<T> {
    type Target = ReadSignal<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> Clone for Signal<T> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

impl<T> Copy for Signal<T> {}

impl<T: 'static> ReadSignal<T> {
    pub fn ref_eq(&self, other: &Self) -> bool {
        self.id == other.id
    }

    pub fn track(&self) {
        RT.with(|rt| {
            if let Some(id) = rt.observer.get() {
                id.with_context(|ctx| ctx.add_dependency(self.id));
            }
        });
    }

    pub fn read<U>(&self, f: impl FnOnce(&T) -> U) -> U {
        self.track();
        self.read_untracked(f)
    }

    pub fn read_untracked<U>(&self, f: impl FnOnce(&T) -> U) -> U {
        RT.with(|rt| {
            let t = rt
                .signals
                .borrow()
                .get(self.id)
                .unwrap_or_else(|| panic!("tried to access a disposed signal"))
                .clone();
            let t = t.borrow();
            f(t.downcast_ref::<T>()
                .unwrap_or_else(|| panic!("tried to use a signal in mismatched types")))
        })
    }

    pub fn get(&self) -> T
    where
        T: Clone,
    {
        self.track();
        self.get_untracked()
    }

    pub fn get_untracked(&self) -> T
    where
        T: Clone,
    {
        self.read_untracked(T::clone)
    }
}

impl<T: 'static> Signal<T> {
    pub fn trigger(&self) {
        self.id.trigger()
    }

    pub fn write(&self, f: impl FnOnce(&mut T)) {
        self.write_slient(f);
        self.trigger();
    }

    pub fn write_slient(&self, f: impl FnOnce(&mut T)) {
        RT.with(|rt| {
            let t = rt
                .signals
                .borrow_mut()
                .get_mut(self.id)
                .unwrap_or_else(|| panic!("tried to access a disposed signal"))
                .clone();
            f(t.borrow_mut()
                .downcast_mut()
                .unwrap_or_else(|| panic!("tried to use a signal in mismatched types")));
        });
    }

    pub fn set(&self, val: T) {
        self.set_slient(val);
        self.trigger();
    }

    pub fn set_slient(&self, val: T) {
        self.write_slient(|v| *v = val);
    }

    pub fn update(&self, f: impl FnOnce(&T) -> T) {
        self.update_silent(f);
        self.trigger();
    }

    pub fn update_silent(&self, f: impl FnOnce(&T) -> T) {
        self.write_slient(|t| *t = f(t));
    }
}

#[derive(Default)]
pub(crate) struct SignalContext {
    subscribers: IndexSet<EffectId, ahash::RandomState>,
}

impl SignalContext {
    pub fn subscribe(&mut self, id: EffectId) {
        self.subscribers.insert(id);
    }

    pub fn unsubscribe(&mut self, id: EffectId) {
        self.subscribers.remove(&id);
    }
}

impl SignalId {
    #[must_use]
    pub fn try_with_context<T>(&self, f: impl FnOnce(&mut SignalContext) -> T) -> Option<T> {
        RT.with(|rt| rt.signal_contexts.borrow_mut().get_mut(*self).map(f))
    }

    pub fn with_context<T>(&self, f: impl FnOnce(&mut SignalContext) -> T) -> T {
        RT.with(|rt| rt.signal_contexts.borrow_mut().get_mut(*self).map(f))
            .unwrap_or_else(|| panic!("tried to access a disposed signal"))
    }

    pub fn trigger(&self) {
        let subscribers = self.with_context(|ctx| {
            ctx.subscribers
                .drain(..)
                .collect::<SmallVec<[_; INITIAL_SUBCRIBER_SLOTS]>>()
        });
        // Effects attach to subscribers at the end of the effect scope, an effect
        // created inside another scope might send signals to its outer scope,
        // so we should ensure the inner effects re-execute before outer ones to
        // avoid potential double executions.
        for id in subscribers {
            // Effect in child scopes may be disposed.
            let _ = id.run();
        }
    }

    pub fn make_signal<T>(&self) -> Signal<T> {
        Signal(ReadSignal {
            id: *self,
            marker: PhantomData,
        })
    }
}

impl Scope {
    fn create_signal_dyn(&self, t: Rc<RefCell<dyn Any>>) -> SignalId {
        RT.with(|rt| {
            let id = rt.signals.borrow_mut().insert(t);
            rt.signal_contexts.borrow_mut().insert(id, <_>::default());
            self.id.on_cleanup(Cleanup::Signal(id));
            id
        })
    }

    pub fn create_signal<T: 'static>(&self, t: T) -> Signal<T> {
        self.create_signal_dyn(Rc::new(RefCell::new(t)))
            .make_signal()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_root;

    #[test]
    fn signal() {
        create_root(|cx| {
            let state = cx.create_signal(0);
            assert_eq!(state.get(), 0);
            state.set(1);
            assert_eq!(state.get(), 1);
        });
    }

    #[test]
    fn signal_composition() {
        create_root(|cx| {
            let state = cx.create_signal(1);
            let double = || state.get() * 2;
            assert_eq!(double(), 2);
            state.set(2);
            assert_eq!(double(), 4);
        });
    }

    #[test]
    fn signal_set_slient() {
        create_root(|cx| {
            let state = cx.create_signal(1);
            let double = cx.create_memo(move || state.get() * 2);
            assert_eq!(double.get(), 2);
            state.set_slient(2);
            assert_eq!(double.get(), 2);
        });
    }

    #[test]
    fn signal_of_signal() {
        create_root(|cx| {
            let state = cx.create_signal(1);
            let state2 = cx.create_signal(state);
            let double = cx.create_memo(move || state2.get().get() * 2);
            assert_eq!(state2.get().get(), 1);
            state.set(2);
            assert_eq!(double.get(), 4);
        });
    }

    struct DropAndRead(Option<Signal<String>>);
    impl Drop for DropAndRead {
        fn drop(&mut self) {
            self.0.unwrap().trigger();
        }
    }

    #[test]
    #[should_panic = "tried to access a disposed signal"]
    fn cannot_read_disposed_signals() {
        create_root(|cx| {
            let var = cx.create_signal(DropAndRead(None));
            let signal = cx.create_signal(String::from("Hello, xFrame!"));
            var.write(|v| v.0 = Some(signal));
        });
    }

    #[test]
    #[should_panic = "tried to access a disposed signal"]
    fn cannot_read_disposed_signal_contexts() {
        create_root(|cx| {
            let var = cx.create_signal(DropAndRead(None));
            let signal = cx.create_signal(String::from("Hello, xFrame!"));
            var.write(|v| v.0 = Some(signal));
        });
    }

    #[test]
    fn access_previous_signal_on_drop() {
        struct DropAndAssert {
            var: Signal<i32>,
            expect: i32,
        }
        impl Drop for DropAndAssert {
            fn drop(&mut self) {
                assert_eq!(self.var.get(), self.expect);
            }
        }

        create_root(|cx| {
            let var = cx.create_signal(777);
            cx.create_signal(DropAndAssert { var, expect: 777 });
        });
    }

    #[test]
    fn fmt_signal() {
        create_root(|cx| {
            let state = cx.create_signal(0);
            assert_eq!(format!("{:?}", state), "Signal(0)");
            assert_eq!(format!("{}", state), "0");
        });
    }
}
