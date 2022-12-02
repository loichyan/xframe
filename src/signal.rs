use crate::{
    scope::{Cleanup, Scope},
    shared::{EffectId, Shared, SignalId, SHARED},
    ThreadLocal,
};
use indexmap::IndexSet;
use smallvec::SmallVec;
use std::{any::Any, fmt, marker::PhantomData, ops::Deref, rc::Rc};

const INITIAL_SUBCRIBER_SLOTS: usize = 4;

pub struct ReadSignal<T: 'static> {
    id: SignalId,
    marker: PhantomData<(T, ThreadLocal)>,
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

pub struct Signal<T: 'static>(ReadSignal<T>);

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
        SHARED.with(|shared| {
            if let Some(id) = shared.observer.get() {
                id.with_context(shared, |ctx| ctx.add_dependency(self.id))
                    .unwrap_or_else(|| unreachable!());
            }
        });
    }

    pub fn read<U>(&self, f: impl FnOnce(&T) -> U) -> U {
        self.track();
        self.read_untracked(f)
    }

    pub fn read_untracked<U>(&self, f: impl FnOnce(&T) -> U) -> U {
        SHARED.with(|shared| {
            f(shared
                .signals
                .borrow()
                .get(self.id)
                .unwrap_or_else(|| panic!("tried to access a disposed signal"))
                .downcast_ref::<T>()
                .unwrap_or_else(|| unreachable!()))
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
        SHARED.with(|shared| self.id.trigger(shared));
    }

    pub fn write<U>(&self, f: impl FnOnce(&mut T) -> U) -> U {
        let output = self.write_slient(f);
        self.trigger();
        output
    }

    pub fn write_slient<U>(&self, f: impl FnOnce(&mut T) -> U) -> U {
        SHARED.with(|shared| {
            f(shared
                .signals
                .borrow_mut()
                .get_mut(self.id)
                .unwrap_or_else(|| panic!("tried to access a disposed signal"))
                .downcast_mut()
                .unwrap_or_else(|| unreachable!()))
        })
    }

    pub fn set(&self, val: T) {
        self.set_slient(val);
        self.trigger();
    }

    pub fn set_slient(&self, val: T) {
        self.write_slient(|v| *v = val);
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
        shared: &Shared,
        f: impl FnOnce(&mut SignalContext) -> T,
    ) -> Option<T> {
        shared.signal_contexts.borrow_mut().get_mut(*self).map(f)
    }

    pub fn trigger(&self, shared: &Shared) {
        let subscribers = self
            .with_context(shared, |ctx| {
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
            id.run(shared);
        }
    }
}

impl Scope {
    fn create_signal_dyn(&self, t: Box<dyn Any>) -> SignalId {
        self.with_shared(|shared| {
            self.id.with(shared, |cx| {
                let id = shared.signals.borrow_mut().insert(t);
                cx.push_cleanup(Cleanup::Signal(id));
                shared
                    .signal_contexts
                    .borrow_mut()
                    .insert(id, <_>::default());
                id
            })
        })
    }

    pub fn create_signal<T>(&self, t: T) -> Signal<T> {
        let id = self.create_signal_dyn(Box::new(t));
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
            let var = cx.create_variable(DropAndRead(None));
            let signal = cx.create_signal(String::from("Hello, xFrame!"));
            var.write(|v| v.0 = Some(signal));
        });
    }

    #[test]
    #[should_panic = "tried to access a disposed signal"]
    fn cannot_read_disposed_signal_contexts() {
        create_root(|cx| {
            let var = cx.create_variable(DropAndRead(None));
            let signal = cx.create_signal(String::from("Hello, xFrame!"));
            var.write(|v| v.0 = Some(signal));
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
