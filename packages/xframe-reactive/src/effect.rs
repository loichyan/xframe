use crate::{
    scope::{BoundedScope, Scope, ScopeShared},
    signal::SignalContext,
    utils::ByAddress,
};
use ahash::AHashSet;
use core::fmt;
use std::cell::RefCell;

#[derive(Debug, Clone, Copy)]
pub struct Effect<'a> {
    inner: &'a RawEffect<'a>,
}

impl<'a> Effect<'a> {
    pub fn run(&self) {
        self.inner.run();
    }
}

#[derive(Debug)]
pub(crate) struct RawEffect<'a> {
    effect: &'a (dyn 'a + AnyEffect),
    shared: ByAddress<'static, ScopeShared>,
    dependencies: RefCell<AHashSet<ByAddress<'a, SignalContext>>>,
}

impl<'a> RawEffect<'a> {
    pub fn add_dependence(&self, signal: &'a SignalContext) {
        self.dependencies.borrow_mut().insert(ByAddress(signal));
    }

    pub fn remove_dependence(&self, signal: &'a SignalContext) {
        self.dependencies.borrow_mut().remove(&ByAddress(signal));
    }

    pub fn clear_dependencies(&self) {
        // SAFETY: this will be dropped after disposing, it's safe to access it.
        let this: &'static RawEffect<'static> = unsafe { std::mem::transmute(&*self) };
        let deps = &mut *self.dependencies.borrow_mut();
        for dep in deps.iter() {
            dep.0.unsubscribe(this);
        }
        deps.clear();
    }

    pub fn run(&self) {
        // SAFETY: A signal might be subscribed by an effect created inside a
        // child scope, calling the effect causes undefined behavior, it's
        // necessary for an effect to notify all its dependencies to unsubscribe
        // itself before it's disposed.
        let this: &'static RawEffect<'static> = unsafe { std::mem::transmute(self) };
        let obs = &self.shared.0.observer;

        // Ignore the recursive effect calls.
        if obs.get().map(ByAddress) == Some(ByAddress(this)) {
            return;
        }

        // Save previous subscriber.
        let saved = obs.take();
        obs.set(Some(this));

        // Re-calculate dependencies.
        self.clear_dependencies();

        // Call the effect.
        self.effect.run();

        // Restore previous subscriber.
        obs.set(saved);
    }
}

impl Drop for RawEffect<'_> {
    fn drop(&mut self) {
        self.clear_dependencies();
    }
}

trait AnyEffect {
    fn run(&self);
}

impl<'a> fmt::Debug for dyn 'a + AnyEffect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<dyn AnyEffect>")
    }
}

struct AnyEffectImpl<T, F> {
    prev: T,
    func: F,
}

impl<T, F> AnyEffect for RefCell<AnyEffectImpl<Option<T>, F>>
where
    F: FnMut(Option<T>) -> T,
{
    fn run(&self) {
        let this = &mut *self.borrow_mut();
        let prev = this.prev.take();
        this.prev = Some((this.func)(prev));
    }
}

fn create_effect_impl<'a>(cx: Scope<'a>, effect: &'a (dyn 'a + AnyEffect)) -> Effect<'a> {
    let inner = cx.create_variable(RawEffect {
        effect,
        shared: ByAddress(cx.shared()),
        dependencies: Default::default(),
    });
    inner.run();
    Effect { inner }
}

impl<'a> Scope<'a> {
    pub fn create_effect<T: 'a>(self, f: impl 'a + FnMut(Option<T>) -> T) -> Effect<'a> {
        let eff = self.create_variable(RefCell::new(AnyEffectImpl {
            prev: None,
            func: f,
        }));
        create_effect_impl(self, eff)
    }

    pub fn create_effect_scoped(
        self,
        mut f: impl 'a + for<'child> FnMut(BoundedScope<'child, 'a>),
    ) -> Effect<'a> {
        self.create_effect(move |disposer| {
            drop(disposer);
            self.create_child(|cx| f(cx))
        })
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
            let state = cx.create_signal(0);
            cx.create_effect(move |_| {
                state.track();
                state.update(|x| *x + 1);
            });
            state.update(|x| *x + 1);
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
    fn remove_a_disposed_dependence() {
        Scope::create_root(|cx| {
            let eff = cx.create_effect(move |prev| {
                if prev.is_none() {
                    cx.create_child(|cx| {
                        let state = cx.create_signal(0);
                        state.track();
                    });
                }
            });
            assert_eq!(eff.inner.dependencies.borrow().len(), 0);
        });
    }
}
