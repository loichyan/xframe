use crate::{
    runtime::{EffectId, HashSet, SignalId, RT},
    scope::{Cleanup, Scope},
    ThreadLocal,
};
use std::{cell::RefCell, fmt, marker::PhantomData, rc::Rc};

pub(crate) type RawEffect = Rc<RefCell<dyn AnyEffect>>;

/// An effect can track signals and automatically execute whenever the captured
/// [`Signal`](crate::signal::Signal)s changed.
#[derive(Clone, Copy)]
pub struct Effect {
    pub(crate) id: EffectId,
    marker: PhantomData<ThreadLocal>,
}

impl fmt::Debug for Effect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Effect").finish_non_exhaustive()
    }
}

impl Effect {
    pub fn run(&self) {
        self.id
            .try_run()
            .unwrap_or_else(|| panic!("tried to access a disposed effect"));
    }
}

#[derive(Default)]
pub(crate) struct EffectContext {
    dependencies: HashSet<SignalId>,
}

impl EffectContext {
    pub fn add_dependency(&mut self, id: SignalId) {
        self.dependencies.insert(id);
    }
}

pub(crate) trait AnyEffect {
    fn run_untracked(&mut self);
}

struct AnyEffectImpl<F>(F);

impl<F> AnyEffect for AnyEffectImpl<F>
where
    F: FnMut(),
{
    fn run_untracked(&mut self) {
        (self.0)();
    }
}

impl EffectId {
    #[must_use]
    fn try_with<T>(&self, f: impl FnOnce(&mut dyn AnyEffect) -> T) -> Option<T> {
        let raw = RT.with(|rt| rt.effects.borrow_mut().get(*self).cloned());
        raw.map(|t| f(&mut *t.borrow_mut()))
    }

    #[must_use]
    pub fn try_with_context<T>(&self, f: impl FnOnce(&mut EffectContext) -> T) -> Option<T> {
        RT.with(|rt| rt.effect_contexts.borrow_mut().get_mut(*self).map(f))
    }

    pub fn with_context<T>(&self, f: impl FnOnce(&mut EffectContext) -> T) -> T {
        self.try_with_context(f)
            .unwrap_or_else(|| panic!("tried to access a disposed effect"))
    }

    #[must_use]
    pub fn try_run(&self) -> Option<()> {
        RT.with(|rt| {
            self.try_with(|effect| {
                // 1) Clear dependencies.
                // After each execution a signal may not be tracked by this effect anymore,
                // so we need to clear dependencies both links and backlinks at first.
                self.with_context(|ctx| {
                    let dependencies = ctx.dependencies.drain();
                    for id in dependencies {
                        // Signal in child scopes may be disposed.
                        let _ = id.try_with_context(|ctx| {
                            ctx.unsubscribe(*self);
                        });
                    }
                });

                // 2) Save observer and change it to this effect.
                let prev = rt.observer.take();
                rt.observer.set(Some(*self));

                // 3) Call the effect.
                effect.run_untracked();

                // 4) Subscribe dependencies.
                // An effect is appended to the subscriber list of the signals since we
                // subscribe after running the closure.
                self.with_context(|ctx| {
                    for id in ctx.dependencies.iter() {
                        let _ = id.try_with_context(|ctx| ctx.subscribe(*self));
                    }
                });

                // 5) Restore previous observer.
                rt.observer.set(prev);
            })
        })
    }
}

impl Scope {
    fn create_effect_dyn(&self, f: Rc<RefCell<dyn AnyEffect>>) -> Effect {
        RT.with(|rt| {
            let id = rt.effects.borrow_mut().insert(f);
            rt.effect_contexts
                .borrow_mut()
                .insert(id, Default::default());
            self.id.on_cleanup(Cleanup::Effect(id));
            id.try_run().unwrap();
            Effect {
                id,
                marker: PhantomData,
            }
        })
    }

    /// Constructs an [`Effect`] which automatically reruns whenever tracked signals update.
    pub fn create_effect(&self, f: impl 'static + FnMut()) -> Effect {
        self.create_effect_dyn(Rc::new(RefCell::new(AnyEffectImpl(f))))
    }

    /// Constructs an [`Effect`] that run with a new child scope each time. The
    /// created [`Scope`] will not be disposed until the next run.
    pub fn create_effect_scoped(&self, mut f: impl 'static + FnMut(Scope)) {
        let cx = *self;
        let mut disposer = None;
        self.create_effect(move || {
            drop(disposer.take());
            disposer = Some(cx.create_child(&mut f));
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_root;

    #[test]
    fn effect() {
        create_root(|cx| {
            let state = cx.create_signal(0);
            let double = cx.create_signal(-1);

            cx.create_effect(move || {
                double.set(state.get() * 2);
            });
            assert_eq!(double.get(), 0);

            state.set(1);
            assert_eq!(double.get(), 2);

            state.set(2);
            assert_eq!(double.get(), 4);
        });
    }

    #[test]
    fn no_infinite_loop_in_effect() {
        create_root(|cx| {
            let state = cx.create_signal(());
            cx.create_effect(move || {
                state.track();
                state.trigger();
            });
            state.trigger();
        });
    }

    #[test]
    fn no_infinite_loop_in_nested_effect() {
        create_root(|cx| {
            let state = cx.create_signal(());
            cx.create_effect({
                let mut first = true;
                move || {
                    if first {
                        state.track();
                        state.trigger();
                    }
                    first = false;
                }
            });
            state.trigger();
        });
    }

    #[test]
    fn dynamically_update_effect_dependencies() {
        create_root(|cx| {
            let cond = cx.create_signal(true);
            let state1 = cx.create_signal(0);
            let state2 = cx.create_signal(0);
            let counter = cx.create_signal(0);

            cx.create_effect(move || {
                counter.update(|x| *x + 1);

                if cond.get() {
                    state1.track();
                } else {
                    state2.track();
                }
            });
            assert_eq!(counter.get(), 1);

            state1.set(1);
            assert_eq!(counter.get(), 2);

            // `state2` is not tracked.
            state2.set(1);
            assert_eq!(counter.get(), 2);

            // `state1` is untracked and `state2` is tracked
            cond.set(false);
            assert_eq!(counter.get(), 3);

            state1.set(2);
            assert_eq!(counter.get(), 3);

            state2.set(2);
            assert_eq!(counter.get(), 4);
        });
    }

    #[test]
    fn inner_effect_triggered_first() {
        create_root(|cx| {
            let state = cx.create_signal(());
            let inner_counter = cx.create_signal(0);
            let outer_counter = cx.create_signal(0);

            cx.create_effect(move || {
                state.track();
                if inner_counter.get_untracked() < 2 {
                    cx.create_effect_scoped(move |cx| {
                        cx.create_effect(move || {
                            state.track();
                            inner_counter.update(|x| x + 1);
                        });
                    });
                }
                outer_counter.update(|x| x + 1);
            });
            assert_eq!(inner_counter.get(), 1);
            assert_eq!(outer_counter.get(), 1);

            // If the outer effect is triggered first, new effects will be created
            // and increase the counters.
            state.trigger();
            assert_eq!(inner_counter.get(), 2);
            assert_eq!(outer_counter.get(), 2);

            state.trigger();
            assert_eq!(inner_counter.get(), 3);
            assert_eq!(outer_counter.get(), 3);
        });
    }

    #[test]
    fn cannot_access_disposed_dependencies() {
        create_root(|cx| {
            let eff = cx.create_effect({
                let mut first = true;
                move || {
                    if first {
                        cx.create_child(|cx| {
                            let state = cx.create_signal(0);
                            state.track();
                        });
                    }
                    first = false;
                }
            });
            let (total, active) = eff.id.with_context(|ctx| {
                (
                    ctx.dependencies.len(),
                    ctx.dependencies
                        .iter()
                        // The `Signal` should be removed after the scope is
                        // disposed.
                        .map(|id| id.try_with_context(|_| ()).is_some())
                        .filter(|x| *x)
                        .count(),
                )
            });
            assert_eq!(total, 1);
            assert_eq!(active, 0);
        });
    }

    #[test]
    #[should_panic = "tried to access a disposed effect"]
    fn cannot_run_disposed_effects() {
        struct DropAndRead(Option<Effect>);
        impl Drop for DropAndRead {
            fn drop(&mut self) {
                self.0.unwrap().run();
            }
        }

        create_root(|cx| {
            let var = cx.create_signal(DropAndRead(None));
            // Last creadted drop first.
            let eff = cx.create_effect(|| ());
            var.write(|v| v.0 = Some(eff));
        });
    }
}
