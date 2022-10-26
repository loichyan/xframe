use crate::scope::{BoundedScope, Cleanup, EffectRef, Scope, Shared, SignalRef};
use ahash::AHashSet;
use std::{cell::RefCell, fmt};

#[derive(Clone, Copy)]
pub struct Effect<'a> {
    inner: &'a RawEffect<'a>,
}

impl fmt::Debug for Effect<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Effect").finish_non_exhaustive()
    }
}

impl<'a> Effect<'a> {
    pub fn run(&self) {
        self.inner.run();
    }
}

pub(crate) struct RawEffect<'a> {
    this: EffectRef,
    shared: &'static Shared,
    effect: &'a (dyn 'a + AnyEffect),
    dependencies: RefCell<AHashSet<SignalRef>>,
}

impl<'a> RawEffect<'a> {
    pub fn add_dependency(&self, signal: SignalRef) {
        self.dependencies.borrow_mut().insert(signal);
    }

    pub fn clear_dependencies(&self, this: EffectRef) {
        let mut dependencies = self.dependencies.borrow_mut();
        for sig in dependencies.iter() {
            sig.with(|sig| sig.unsubscribe(this));
        }
        dependencies.clear();
    }

    pub fn run(&self) {
        // 1) Clear dependencies.
        self.clear_dependencies(self.this);

        // 2) Save previous subscriber.
        let observer = &self.shared.observer;
        let saved = observer.take();
        observer.set(Some(self.this));

        // 3) Call the effect.
        self.effect.run_untracked();

        // 4) Re-calculate dependencies.
        for sig in self.dependencies.borrow().iter() {
            sig.with(|sig| sig.subscribe(self.this));
        }

        // 5) Restore previous subscriber.
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

impl<'a> Scope<'a> {
    fn create_effect_impl(self, effect: &'a (dyn 'a + AnyEffect)) -> Effect<'a> {
        let shared = self.shared();
        let raw = shared.effects.alloc_with_weak(|this| {
            let raw = RawEffect {
                this,
                effect,
                shared,
                dependencies: Default::default(),
            };
            // SAFETY: same as what we do on RawSignal, the effect cannot be accessed
            // once this scope is disposed.
            unsafe { std::mem::transmute(raw) }
        });
        let inner = unsafe { raw.leak_ref() };
        self.push_cleanup(Cleanup::Effect(raw));
        inner.run();
        Effect { inner }
    }

    pub fn create_effect<T: 'a>(self, f: impl 'a + FnMut(Option<T>) -> T) -> Effect<'a> {
        let effect = self.create_variable(RefCell::new(AnyEffectImpl {
            prev: None,
            func: f,
        }));
        self.create_effect_impl(effect)
    }

    pub fn create_effect_scoped(
        self,
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
    use std::cell::Cell;

    #[test]
    fn reactive_effect() {
        Scope::create_root(|cx| {
            let state = cx.create_signal(0);
            let double = cx.create_variable(Cell::new(-1));

            cx.create_effect(move |_| {
                double.set(*state.get() * 2);
            });
            assert_eq!(double.get(), 0);

            state.set(1);
            assert_eq!(double.get(), 2);

            state.set(2);
            assert_eq!(double.get(), 4);
        });
    }

    #[test]
    fn previous_returned_value_in_effect() {
        Scope::create_root(|cx| {
            let state = cx.create_signal(0);
            let prev_state = cx.create_signal(-1);

            cx.create_effect(move |prev| {
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
        Scope::create_root(|cx| {
            let state = cx.create_signal(());
            cx.create_effect(move |_| {
                state.track();
                state.trigger_subscribers();
            });
            state.trigger_subscribers();
        });
    }

    #[test]
    fn no_infinite_loop_in_nested_effect() {
        Scope::create_root(|cx| {
            let state = cx.create_signal(());
            cx.create_effect(move |prev| {
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
        Scope::create_root(|cx| {
            let cond = cx.create_signal(true);

            let state1 = cx.create_signal(0);
            let state2 = cx.create_signal(0);

            let counter = cx.create_signal(0);

            cx.create_effect(move |_| {
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
        Scope::create_root(|cx| {
            let state = cx.create_signal(());
            let inner_counter = cx.create_variable(Cell::new(0));
            let outer_counter = cx.create_variable(Cell::new(0));

            cx.create_effect(move |_| {
                state.track();
                if inner_counter.get() < 2 {
                    cx.create_effect_scoped(move |cx| {
                        cx.create_effect(move |_| {
                            state.track();
                            inner_counter.set(inner_counter.get() + 1);
                        });
                    });
                }
                outer_counter.set(outer_counter.get() + 1);
            });
            assert_eq!(inner_counter.get(), 1);
            assert_eq!(outer_counter.get(), 1);

            state.trigger_subscribers();
            assert_eq!(inner_counter.get(), 2);
            assert_eq!(outer_counter.get(), 2);

            state.trigger_subscribers();
            assert_eq!(inner_counter.get(), 3);
            assert_eq!(outer_counter.get(), 3);
        });
    }

    #[test]
    fn remove_a_disposed_dependency() {
        Scope::create_root(|cx| {
            let eff = cx.create_effect(move |prev| {
                if prev.is_none() {
                    cx.create_child(|cx| {
                        let state = cx.create_signal(0);
                        state.track();
                    });
                }
            });
            let count = eff
                .inner
                .dependencies
                .borrow()
                .iter()
                .map(|sig| sig.can_upgrade())
                .filter(|x| *x)
                .count();
            assert_eq!(count, 0);
        });
    }
}
