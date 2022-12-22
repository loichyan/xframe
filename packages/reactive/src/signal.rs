use crate::{
    runtime::{EffectId, Runtime, SignalId, RT},
    scope::{Cleanup, Scope},
    ThreadLocal,
};
use indexmap::IndexSet;
use smallvec::SmallVec;
use std::{any::Any, cell::RefCell, fmt, marker::PhantomData, ops::Deref, rc::Rc};

const INITIAL_SUBCRIBER_SLOTS: usize = 4;

pub(crate) type RawSignal = Rc<RefCell<dyn Any>>;

pub struct ReadSignal<T: 'static> {
    pub(crate) id: SignalId,
    pub(crate) marker: PhantomData<(T, ThreadLocal)>,
}

impl<T: fmt::Debug> fmt::Debug for ReadSignal<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.read(|v| f.debug_tuple("Signal").field(v).finish())
    }
}

impl<T: fmt::Display> fmt::Display for ReadSignal<T> {
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

pub struct Signal<T: 'static>(pub(crate) ReadSignal<T>);

impl<T: fmt::Debug> fmt::Debug for Signal<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.deref().fmt(f)
    }
}

impl<T: fmt::Display> fmt::Display for Signal<T> {
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

impl<T> ReadSignal<T> {
    pub fn track(&self) {
        RT.with(|rt| {
            if let Some(id) = rt.observer.get() {
                id.with_context(rt, |ctx| ctx.add_dependency(self.id))
                    .unwrap();
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

impl<T> Signal<T> {
    pub fn trigger(&self) {
        RT.with(|rt| self.id.trigger(rt));
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
    pub fn with_context<T>(
        &self,
        rt: &Runtime,
        f: impl FnOnce(&mut SignalContext) -> T,
    ) -> Option<T> {
        rt.signal_contexts.borrow_mut().get_mut(*self).map(f)
    }

    pub fn trigger(&self, rt: &Runtime) {
        let subscribers = self
            .with_context(rt, |ctx| {
                ctx.subscribers
                    .drain(..)
                    .collect::<SmallVec<[_; INITIAL_SUBCRIBER_SLOTS]>>()
            })
            .unwrap_or_else(|| panic!("tried to access a disposed signal"));
        // Effects attach to subscribers at the end of the effect scope, an effect
        // created inside another scope might send signals to its outer scope,
        // so we should ensure the inner effects re-execute before outer ones to
        // avoid potential double executions.
        for id in subscribers {
            id.run(rt);
        }
    }
}

impl Scope {
    fn create_signal_dyn(&self, t: Rc<RefCell<dyn Any>>) -> SignalId {
        RT.with(|rt| {
            self.id.with(rt, |cx| {
                let id = rt.signals.borrow_mut().insert(t);
                cx.push_cleanup(Cleanup::Signal(id));
                rt.signal_contexts.borrow_mut().insert(id, <_>::default());
                id
            })
        })
    }

    pub fn create_signal<T>(&self, t: T) -> Signal<T> {
        let id = self.create_signal_dyn(Rc::new(RefCell::new(t)));
        Signal(ReadSignal {
            id,
            marker: PhantomData,
        })
    }

    pub fn create_signal_rc<T>(&self, t: T) -> Signal<Rc<T>> {
        self.create_signal(Rc::new(t))
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
