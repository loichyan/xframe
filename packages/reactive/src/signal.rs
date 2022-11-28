mod modify;
pub use modify::SignalModify;

use crate::{
    scope::{Cleanup, Scope},
    shared::{EffectId, Shared, SignalId, SHARED},
    variable::{VarRef, VarRefMut, VarSlot},
    CovariantLifetime, Empty,
};
use indexmap::IndexSet;
use std::{marker::PhantomData, ops::Deref, ptr::NonNull};

pub struct ReadSignal<'a, T> {
    id: SignalId,
    marker: PhantomData<(T, CovariantLifetime<'a>)>,
}

impl<T> Clone for ReadSignal<'_, T> {
    fn clone(&self) -> Self {
        Self { ..*self }
    }
}

impl<T> Copy for ReadSignal<'_, T> {}

pub struct Signal<'a, T>(ReadSignal<'a, T>);

impl<'a, T> Deref for Signal<'a, T> {
    type Target = ReadSignal<'a, T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> Clone for Signal<'_, T> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

impl<T> Copy for Signal<'_, T> {}

impl<'a, T> ReadSignal<'a, T> {
    fn value(&self) -> &'a VarSlot<T> {
        SHARED.with(|shared| {
            let ptr = shared
                .signals
                .borrow()
                .get(self.id)
                .copied()
                .unwrap_or_else(|| panic!("tried to access a disposed signal"));
            // SAFETY: The type is assumed by the `ty` marker and the value lives
            // as long as current `Scope`.
            unsafe { ptr.cast().as_ref() }
        })
    }

    pub fn track(&self) {
        SHARED.with(|shared| {
            if let Some(id) = shared.observer.get() {
                id.with_context(shared, |ctx| ctx.add_dependency(self.id))
                    .unwrap_or_else(|| unreachable!());
            }
        });
    }

    pub fn get(&self) -> VarRef<'_, T> {
        self.track();
        self.get_untracked()
    }

    pub fn get_untracked(&self) -> VarRef<'_, T> {
        self.value().get()
    }
}

impl<'a, T> Signal<'a, T> {
    pub fn trigger(&self) {
        SHARED.with(|shared| self.id.trigger(shared));
    }

    pub fn get_mut(&self) -> VarRefMut<'_, T> {
        self.value().get_mut()
    }

    pub fn set(&self, val: T) {
        self.set_slient(val);
        self.trigger();
    }

    pub fn set_slient(&self, val: T) {
        *self.value().get_mut() = val;
    }

    pub fn update(&self, f: impl FnOnce(&mut T) -> T) {
        self.update_slient(f);
        self.trigger();
    }

    pub fn update_slient(&self, f: impl FnOnce(&mut T) -> T) {
        let mut mut_borrow = self.value().get_mut();
        *mut_borrow = f(&mut *mut_borrow);
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
                let new = IndexSet::with_capacity_and_hasher(ctx.subscribers.len(), <_>::default());
                std::mem::replace(&mut ctx.subscribers, new)
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

impl<'a> Scope<'a> {
    pub fn create_signal<T: 'a>(&self, t: T) -> Signal<'a, T> {
        self.with_shared(|shared| {
            self.id.with(shared, |cx| {
                let value = {
                    // SAFETY: Same as creating variables.
                    unsafe {
                        let ptr = cx.alloc_var(t);
                        std::mem::transmute(NonNull::from(ptr as &dyn Empty))
                    }
                };
                let id = shared.signals.borrow_mut().insert(value);
                shared
                    .signal_contexts
                    .borrow_mut()
                    .insert(id, <_>::default());
                cx.add_cleanup(Cleanup::Signal(id));
                Signal(ReadSignal {
                    id,
                    marker: PhantomData,
                })
            })
        })
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
            assert_eq!(*state.get(), 0);
            state.set(1);
            assert_eq!(*state.get(), 1);
        });
    }

    #[test]
    fn signal_composition() {
        create_root(|cx| {
            let state = cx.create_signal(1);
            let double = || *state.get() * 2;
            assert_eq!(double(), 2);
            state.set(2);
            assert_eq!(double(), 4);
        });
    }

    #[test]
    fn signal_set_slient() {
        create_root(|cx| {
            let state = cx.create_signal(1);
            let double = cx.create_memo(move || *state.get() * 2);
            assert_eq!(*double.get(), 2);
            state.set_slient(2);
            assert_eq!(*double.get(), 2);
        });
    }

    #[test]
    fn signal_of_signal() {
        create_root(|cx| {
            let state = cx.create_signal(1);
            let state2 = cx.create_signal(state);
            let double = cx.create_memo(move || *state2.get().get() * 2);
            assert_eq!(*state2.get().get(), 1);
            state.set(2);
            assert_eq!(*double.get(), 4);
        });
    }

    struct DropAndRead<'a>(Option<Signal<'a, String>>);
    impl Drop for DropAndRead<'_> {
        fn drop(&mut self) {
            self.0.unwrap().trigger();
        }
    }

    #[test]
    #[should_panic = "tried to access a disposed signal"]
    fn cannot_read_a_disposed_signal_value() {
        create_root(|cx| {
            let var = cx.create_variable(DropAndRead(None));
            let signal = cx.create_signal(String::from("Hello, xFrame!"));
            var.get_mut().0 = Some(signal);
        });
    }

    #[test]
    #[should_panic = "tried to access a disposed signal"]
    fn cannot_read_a_disposed_signal_context() {
        create_root(|cx| {
            let var = cx.create_variable(DropAndRead(None));
            let signal = cx.create_signal(String::from("Hello, xFrame!"));
            var.get_mut().0 = Some(signal);
        });
    }
}
