use crate::scope::{BoundedScope, Cleanup, EffectId, Scope, Shared, SignalId};
use ahash::AHashSet;
use std::{cell::RefCell, fmt};

#[derive(Clone, Copy)]
pub struct Effect<'a> {
    inner: RawEffect<'a>,
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

pub(crate) type RawEffect<'a> = &'a (dyn 'a + AnyEffect);

pub(crate) trait AnyEffect {
    fn this(&self) -> EffectId;
    fn shared(&self) -> &Shared;
    fn dependencies(&self) -> &RefCell<AHashSet<SignalId>>;
    fn run_effect(&self);
}

impl<'a> dyn 'a + AnyEffect {
    pub fn add_dependency(&self, id: SignalId) {
        self.dependencies().borrow_mut().insert(id);
    }

    pub fn run(&self) {
        // 1) Clear dependencies.
        let this = self.this();
        let shared = self.shared();
        let dependencies = self.dependencies();
        {
            let mut deps = dependencies.borrow_mut();
            for id in deps.iter().copied() {
                shared
                    .signals
                    .borrow()
                    .get(id)
                    .map(|sig| sig.unsubscribe(this));
            }
            deps.clear();
        }

        // 2) Save previous subscriber.
        let observer = &shared.observer;
        let saved = observer.take();
        observer.set(Some(this));

        // 3) Call the effect.
        self.run_effect();

        // 4) Re-calculate dependencies.
        for id in dependencies.borrow().iter().copied() {
            shared
                .signals
                .borrow()
                .get(id)
                .map(|sig| sig.subscribe(this));
        }

        // 5) Restore previous subscriber.
        observer.set(saved);
    }
}

struct AnyEffectImpl<'a, T, F> {
    this: EffectId,
    shared: &'a Shared,
    prev: RefCell<Option<T>>,
    func: RefCell<F>,
    dependencies: RefCell<AHashSet<SignalId>>,
}

impl<'a, T, F> AnyEffect for AnyEffectImpl<'a, T, F>
where
    F: FnMut(Option<T>) -> T,
{
    fn this(&self) -> EffectId {
        self.this
    }

    fn shared(&self) -> &Shared {
        self.shared
    }

    fn dependencies(&self) -> &RefCell<AHashSet<SignalId>> {
        &self.dependencies
    }

    fn run_effect(&self) {
        let mut prev = self.prev.borrow_mut();
        *prev = Some((self.func.borrow_mut())(prev.take()));
    }
}

impl<'a> Scope<'a> {
    fn create_effect_impl(self, effect: &'a (dyn 'a + AnyEffect)) -> Effect<'a> {
        self.push_cleanup(Cleanup::Effect(effect.this()));
        effect.run();
        Effect { inner: effect }
    }

    pub fn create_effect<T: 'a>(self, f: impl 'a + FnMut(Option<T>) -> T) -> Effect<'a> {
        let shared = self.shared();
        let mut effect = None;
        shared.effects.borrow_mut().insert_with_key(|id| {
            let any_impl = AnyEffectImpl {
                this: id,
                shared,
                prev: None.into(),
                func: f.into(),
                dependencies: Default::default(),
            };
            let any = self.create_variable(any_impl) as &dyn AnyEffect;
            effect = Some(any);
            unsafe { std::mem::transmute(any) }
        });
        self.create_effect_impl(effect.unwrap_or_else(|| unreachable!()))
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
                .dependencies()
                .borrow()
                .iter()
                .copied()
                .map(|id| cx.shared().signals.borrow().get(id).is_some())
                .filter(|x| *x)
                .count();
            assert_eq!(count, 0);
        });
    }
}
