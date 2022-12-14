use crate::{
    shared::{EffectId, ScopeId, Shared, SignalId, VariableId, SHARED},
    ThreadLocal,
};
use smallvec::SmallVec;
use std::{fmt, marker::PhantomData, rc::Rc};

const INITIAL_CLEANUP_SLOTS: usize = 4;

#[derive(Clone, Copy)]
pub struct Scope {
    pub(crate) id: ScopeId,
    marker: PhantomData<ThreadLocal>,
}

impl fmt::Debug for Scope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.with_shared(|shared| {
            self.id.with(shared, |raw| {
                f.debug_struct("Scope")
                    .field("cleanups", &raw.cleanups as _)
                    .finish_non_exhaustive()
            })
        })
    }
}

#[derive(Default)]
pub(crate) struct RawScope {
    cleanups: SmallVec<[Cleanup; INITIAL_CLEANUP_SLOTS]>,
}

pub(crate) enum Cleanup {
    Effect(EffectId),
    Signal(SignalId),
    Variable(VariableId),
    Callback(Box<(dyn FnOnce())>),
}

impl fmt::Debug for Cleanup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Cleanup::Effect(_) => f.debug_tuple("Effect"),
            Cleanup::Signal(_) => f.debug_tuple("Signal"),
            Cleanup::Variable(_) => f.debug_tuple("Variable"),
            Cleanup::Callback(_) => f.debug_tuple("Callback"),
        }
        .field(&"_")
        .finish()
    }
}

pub struct ScopeDisposer(Scope);

impl ScopeDisposer {
    fn new(parent: Option<ScopeId>) -> Self {
        SHARED.with(|shared| {
            let id = shared.scopes.borrow_mut().insert(<_>::default());
            if let Some(parent) = parent {
                shared.scope_parents.borrow_mut().insert(id, parent);
            }
            let scope = Scope {
                id,
                marker: PhantomData,
            };
            Self(scope)
        })
    }

    /// Consumes and leaks the [`ScopeDisposer`], returning a [`Scope`] with
    /// same lifetime.
    pub fn leak(self) -> Scope {
        let cx = self.0;
        std::mem::forget(self);
        cx
    }

    /// Constructs a [`ScopeDisposer`] from the leaked [`Scope`].
    pub fn from_leaked(cx: Scope) -> Self {
        Self(cx)
    }
}

impl Drop for ScopeDisposer {
    fn drop(&mut self) {
        SHARED.with(|shared| {
            let Scope { id, .. } = self.0;
            // 1) Cleanup resources created inside this `Scope`.
            let cleanups = id.with(shared, |cx| std::mem::take(&mut cx.cleanups));
            for cl in cleanups.into_iter().rev() {
                match cl {
                    Cleanup::Signal(id) => {
                        let v = shared
                            .signals
                            .borrow_mut()
                            .remove(id)
                            .unwrap_or_else(|| unreachable!());
                        shared
                            .signal_contexts
                            .borrow_mut()
                            .remove(id)
                            .unwrap_or_else(|| unreachable!());
                        drop(v);
                    }
                    Cleanup::Effect(id) => {
                        let v = shared
                            .effects
                            .borrow_mut()
                            .remove(id)
                            .unwrap_or_else(|| unreachable!());
                        shared
                            .effect_contexts
                            .borrow_mut()
                            .remove(id)
                            .unwrap_or_else(|| unreachable!());
                        if Rc::strong_count(&v) != 1 {
                            panic!("tried to dispose an effect in use")
                        }
                        drop(v);
                    }
                    Cleanup::Variable(id) => {
                        let v = shared
                            .variables
                            .borrow_mut()
                            .remove(id)
                            .unwrap_or_else(|| unreachable!());
                        drop(v);
                    }
                    Cleanup::Callback(cb) => shared.untrack(cb),
                }
            }
            // 2) Cleanup resources onwed by this `Scope`.
            shared
                .scopes
                .borrow_mut()
                .remove(id)
                .unwrap_or_else(|| unreachable!());
            shared.scope_contexts.borrow_mut().remove(id);
            shared.scope_parents.borrow_mut().remove(id);
        });
    }
}

impl RawScope {
    pub fn push_cleanup(&mut self, cleanup: Cleanup) {
        self.cleanups.push(cleanup);
    }
}

impl Shared {
    pub fn untrack(&self, f: impl FnOnce()) {
        let prev = self.observer.take();
        f();
        self.observer.set(prev);
    }
}

impl ScopeId {
    pub fn with<T>(&self, shared: &Shared, f: impl FnOnce(&mut RawScope) -> T) -> T {
        shared
            .scopes
            .borrow_mut()
            .get_mut(*self)
            .map(f)
            .unwrap_or_else(|| panic!("tried to access a disposed scope"))
    }
}

pub fn create_root(f: impl FnOnce(Scope)) -> ScopeDisposer {
    let disposer = ScopeDisposer::new(None);
    f(disposer.0);
    disposer
}

impl Scope {
    pub(crate) fn with_shared<T>(&self, f: impl FnOnce(&Shared) -> T) -> T {
        SHARED.with(f)
    }

    pub fn create_child(&self, f: impl FnOnce(Scope)) -> ScopeDisposer {
        let disposer = ScopeDisposer::new(Some(self.id));
        f(disposer.0);
        disposer
    }

    pub fn untrack(&self, f: impl FnOnce()) {
        self.with_shared(|shared| shared.untrack(f));
    }

    pub fn on_cleanup(&self, f: impl 'static + FnOnce()) {
        self.with_shared(|shared| {
            self.id.with(shared, |cx| {
                cx.cleanups.push(Cleanup::Callback(Box::new(f)));
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        cell::{Cell, RefCell},
        rc::Rc,
    };

    #[test]
    fn cleanup() {
        create_root(|cx| {
            let called = cx.create_signal(0);
            cx.create_child(move |cx| {
                cx.on_cleanup(move || {
                    called.update(|x| x + 1);
                });
            });
            assert_eq!(called.get(), 1);
        });
    }

    #[test]
    fn untrack() {
        create_root(|cx| {
            let trigger2 = cx.create_signal(());
            let trigger1 = cx.create_signal(());
            let counter = cx.create_signal(0);

            cx.create_effect(move || {
                trigger1.track();
                cx.untrack(|| {
                    trigger2.track();
                    counter.update(|x| x + 1);
                });
            });
            assert_eq!(counter.get(), 1);

            trigger1.trigger();
            assert_eq!(counter.get(), 2);

            trigger2.trigger();
            assert_eq!(counter.get(), 2);
        });
    }

    #[test]
    fn cleanup_is_untracked() {
        create_root(|cx| {
            let trigger_effect = cx.create_signal(());
            let trigger_cleanup = cx.create_signal(());
            let counter = cx.create_signal(0);

            cx.create_effect_scoped(move |cx| {
                trigger_effect.track();
                counter.update(|x| x + 1);

                cx.on_cleanup(move || {
                    trigger_cleanup.track();
                });
            });
            assert_eq!(counter.get(), 1);

            // Trigger effect to let the scope be disposed and run the cleanup.
            trigger_effect.trigger();
            assert_eq!(counter.get(), 2);

            // This should not work since the cleanup is untracked.
            trigger_cleanup.trigger();
            assert_eq!(counter.get(), 2);
        });
    }

    #[test]
    fn cleanup_in_scoped_effect() {
        create_root(|cx| {
            let trigger = cx.create_signal(());
            let counter = cx.create_signal(0);

            cx.create_effect_scoped(move |cx| {
                trigger.track();
                cx.on_cleanup(move || {
                    counter.update(|x| x + 1);
                });
            });
            assert_eq!(counter.get(), 0);

            trigger.trigger();
            assert_eq!(counter.get(), 1);

            // A new scope is disposed.
            trigger.trigger();
            assert_eq!(counter.get(), 2);
        });
    }

    #[test]
    fn leak_scope() {
        thread_local! {
            static COUNTER: Cell<usize> = Cell::new(0);
        }
        struct DropAndInc;
        impl Drop for DropAndInc {
            fn drop(&mut self) {
                COUNTER.with(|x| x.set(x.get() + 1));
            }
        }
        let disposer = create_root(|_| {});
        let cx = disposer.leak();
        cx.create_signal(DropAndInc);
        drop(ScopeDisposer::from_leaked(cx));
        assert_eq!(COUNTER.with(Cell::get), 1);
    }

    #[test]
    #[should_panic = "tried to dispose an effect in use"]
    fn cannot_dispose_effects_in_use() {
        create_root(|cx| {
            let trigger = cx.create_signal(());
            let disposer = Rc::new(RefCell::new(None));
            let child = {
                let disposer = disposer.clone();
                cx.create_child(move |cx| {
                    cx.create_effect(move || {
                        trigger.track();
                        if let Some(disposer) = disposer.take() {
                            // Current effect is running, this panics to dispose it.
                            drop(disposer);
                        }
                    });
                })
            };
            disposer.replace(Some(child));
            trigger.trigger();
        });
    }
}
