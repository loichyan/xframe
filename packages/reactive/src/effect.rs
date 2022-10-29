use crate::{
    scope::{BoundedScope, Cleanup, OwnedScope},
    shared::{CovariantLifetime, EffectRef, Shared, SignalContextRef},
};
use ahash::AHashSet;
use std::{cell::RefCell, fmt, marker::PhantomData};

pub type Effect<'a> = &'a OwnedEffect<'a>;

impl fmt::Debug for OwnedEffect<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Effect").finish_non_exhaustive()
    }
}

/// An effect can track signals and automatically execute whenever the captured
/// signals changed.
pub struct OwnedEffect<'a> {
    raw: EffectRef,
    shared: &'static Shared,
    bounds: PhantomData<CovariantLifetime<'a>>,
}

impl<'a> OwnedEffect<'a> {
    pub fn run(&self) {
        self.raw.get().run(self.raw, self.shared);
    }
}

pub(crate) struct RawEffect<'a> {
    effect: &'a (dyn 'a + AnyEffect),
    dependencies: RefCell<AHashSet<SignalContextRef>>,
}

impl<'a> RawEffect<'a> {
    pub fn add_dependency(&self, signal: SignalContextRef) {
        self.dependencies.borrow_mut().insert(signal);
    }

    pub fn clear_dependencies(&self, this: EffectRef) {
        let dependencies = &mut *self.dependencies.borrow_mut();
        for sig in dependencies.iter() {
            sig.try_get().map(|sig| sig.unsubscribe(this));
        }
        dependencies.clear();
    }

    pub fn run(&self, this: EffectRef, shared: &Shared) {
        // 1) Clear dependencies.
        // After each execution a signal may not be tracked by this effect anymore,
        // so we need to clear dependencies both links and backlinks at first.
        self.clear_dependencies(this);

        // 2) Save previous observer.
        let observer = &shared.observer;
        let saved = observer.take();
        observer.set(Some(this));

        // 3) Call the effect.
        self.effect.run_untracked();

        // 4) Subscribe dependencies.
        // An effect is appended to the subscriber list of the signals since we
        // subscribe after running the closure.
        for sig in self.dependencies.borrow().iter() {
            sig.try_get().map(|sig| sig.subscribe(this));
        }

        // 5) Restore previous observer.
        observer.set(saved);
    }
}

trait AnyEffect {
    fn run_untracked(&self);
}

struct AnyEffectImpl<T, F> {
    prev: Option<T>,
    func: F,
}

impl<T, F> AnyEffect for RefCell<AnyEffectImpl<T, F>>
where
    F: FnMut(Option<T>) -> T,
{
    fn run_untracked(&self) {
        let effect = &mut *self.borrow_mut();
        let prev = effect.prev.take();
        effect.prev = Some((effect.func)(prev));
    }
}

impl<'a> OwnedScope<'a> {
    fn create_owned_effect(&'a self, effect: &'a (dyn 'a + AnyEffect)) -> OwnedEffect<'a> {
        let shared = self.shared();
        let raw = shared.effects.alloc({
            let raw = RawEffect {
                effect,
                dependencies: Default::default(),
            };
            // SAFETY: Same as `create_signal_context`, the effect cannot be
            // accessed once this scope is disposed.
            unsafe { std::mem::transmute(raw) }
        });
        self.push_cleanup(Cleanup::Effect(raw));
        raw.get().run(raw, shared);
        OwnedEffect {
            raw,
            shared,
            bounds: PhantomData,
        }
    }

    fn create_effect_impl(&'a self, effect: &'a (dyn 'a + AnyEffect)) -> Effect<'a> {
        let owned = self.create_owned_effect(effect);
        unsafe { self.create_variable_unchecked(owned) }
    }

    /// Create an effect which accepts its previous returned value as the parameter
    /// and automatically reruns whenever tracked signals update.
    pub fn create_effect<T: 'a>(&'a self, f: impl 'a + FnMut(Option<T>) -> T) -> Effect<'a> {
        // SAFETY: An effect itself will never access captured variables while
        // being disposed.
        let effect = unsafe {
            self.create_variable_unchecked(RefCell::new(AnyEffectImpl {
                prev: None,
                func: f,
            }))
        };
        self.create_effect_impl(effect)
    }

    /// Create an effect that run with a new child scope each time. The created
    /// scope will not be disposed until the next run.
    pub fn create_effect_scoped(
        &'a self,
        mut f: impl 'a + for<'child> FnMut(BoundedScope<'child, 'a>),
    ) {
        self.create_effect(move |disposer| {
            drop(disposer);
            self.create_child(|cx| f(cx))
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_root;

    #[test]
    fn reactive_effect() {
        create_root(|cx| {
            let state = cx.create_signal(0);
            let double = cx.create_variable(-1);

            cx.create_effect(|_| {
                *double.get_mut() = *state.get() * 2;
            });
            assert_eq!(*double.get(), 0);

            state.set(1);
            assert_eq!(*double.get(), 2);

            state.set(2);
            assert_eq!(*double.get(), 4);
        });
    }

    #[test]
    fn previous_returned_value_in_effect() {
        create_root(|cx| {
            let state = cx.create_signal(0);
            let prev_state = cx.create_signal(-1);

            cx.create_effect(|prev| {
                if let Some(prev) = prev {
                    prev_state.set(prev)
                }
                *state.get()
            });
            assert_eq!(*prev_state.get(), -1);

            state.set(1);
            assert_eq!(*prev_state.get(), 0);

            state.set(2);
            assert_eq!(*prev_state.get(), 1);
        });
    }

    #[test]
    fn no_infinite_loop_in_effect() {
        create_root(|cx| {
            let state = cx.create_signal(());
            cx.create_effect(|_| {
                state.track();
                state.trigger_subscribers();
            });
            state.trigger_subscribers();
        });
    }

    #[test]
    fn no_infinite_loop_in_nested_effect() {
        create_root(|cx| {
            let state = cx.create_signal(());
            cx.create_effect(|prev| {
                if prev.is_none() {
                    state.track();
                    state.trigger_subscribers();
                }
            });
            state.trigger_subscribers();
        });
    }

    #[test]
    fn dynamically_update_effect_dependencies() {
        create_root(|cx| {
            let cond = cx.create_signal(true);

            let state1 = cx.create_signal(0);
            let state2 = cx.create_signal(0);

            let counter = cx.create_signal(0);

            cx.create_effect(|_| {
                counter.update(|x| *x + 1);

                if *cond.get() {
                    state1.track();
                } else {
                    state2.track();
                }
            });
            assert_eq!(*counter.get(), 1);

            state1.set(1);
            assert_eq!(*counter.get(), 2);

            state2.set(1);
            assert_eq!(*counter.get(), 2);

            cond.set(false);
            assert_eq!(*counter.get(), 3);

            state1.set(2);
            assert_eq!(*counter.get(), 3);

            state2.set(2);
            assert_eq!(*counter.get(), 4);
        });
    }

    #[test]
    fn inner_effect_triggered_first() {
        create_root(|cx| {
            let state = cx.create_signal(());
            let inner_counter = cx.create_variable(0);
            let outer_counter = cx.create_variable(0);

            cx.create_effect(|_| {
                state.track();
                if *inner_counter.get() < 2 {
                    cx.create_effect_scoped(|cx| {
                        cx.create_effect(|_| {
                            state.track();
                            *inner_counter.get_mut() += 1;
                        });
                    });
                }
                *outer_counter.get_mut() += 1;
            });
            assert_eq!(*inner_counter.get(), 1);
            assert_eq!(*outer_counter.get(), 1);

            state.trigger_subscribers();
            assert_eq!(*inner_counter.get(), 2);
            assert_eq!(*outer_counter.get(), 2);

            state.trigger_subscribers();
            assert_eq!(*inner_counter.get(), 3);
            assert_eq!(*outer_counter.get(), 3);
        });
    }

    #[test]
    fn remove_a_disposed_dependency() {
        create_root(|cx| {
            let eff = cx.create_effect(|prev| {
                if prev.is_none() {
                    cx.create_child(|cx| {
                        let state = cx.create_signal(0);
                        state.track();
                    });
                }
            });
            let total = eff.raw.get().dependencies.borrow().len();
            assert_eq!(total, 1);
            let active = eff
                .raw
                .get()
                .dependencies
                .borrow()
                .iter()
                .map(|sig| sig.can_upgrade())
                .filter(|x| *x)
                .count();
            assert_eq!(active, 0);
        });
    }

    #[test]
    #[should_panic = "get a disposed slot"]
    fn cannot_read_run_a_disposed_effect() {
        struct DropAndRead<'a>(Option<Effect<'a>>);
        impl Drop for DropAndRead<'_> {
            fn drop(&mut self) {
                self.0.unwrap().run();
            }
        }

        create_root(|cx| {
            let var = cx.create_variable(DropAndRead(None));
            let eff = cx.create_effect(|_| ());
            var.get_mut().0 = Some(eff);
        });
    }
}
