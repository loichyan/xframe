use crate::{
    shared::{EffectId, ScopeId, Shared, SignalId, VariableId, SHARED},
    variable::VarSlot,
    CovariantLifetime, Empty, InvariantLifetime,
};
use bumpalo::Bump;
use smallvec::SmallVec;
use std::{cell::Cell, marker::PhantomData, ptr::NonNull};

const INITIAL_CLEANUP_SLOTS: usize = 4;

pub type Scope<'a> = BoundedScope<'a, 'a>;

#[derive(Clone, Copy)]
pub struct BoundedScope<'a, 'b: 'a> {
    pub(crate) id: ScopeId,
    marker: PhantomData<(InvariantLifetime<'a>, CovariantLifetime<'b>)>,
}

#[derive(Default)]
pub(crate) struct RawScope {
    arena: Bump,
    cleanups: SmallVec<[Cleanup; INITIAL_CLEANUP_SLOTS]>,
}

#[derive(Clone, Copy)]
pub(crate) enum Cleanup {
    Effect(EffectId),
    Signal(SignalId),
    Variable(VariableId),
    Callback(NonNull<dyn Callback>),
}

/// Helper trait to invoke [`FnOnce`].
///
/// Original post: <https://users.rust-lang.org/t/invoke-mut-dyn-fnonce/59356/4>
pub(crate) trait Callback {
    /// Please make sure this will only be called once!
    unsafe fn call_once(&mut self, shared: &Shared);
}

impl<F: FnOnce()> Callback for F {
    unsafe fn call_once(&mut self, shared: &Shared) {
        shared.untrack(|| std::ptr::read(self)())
    }
}

pub struct ScopeDisposer<'a>(Scope<'a>);

impl<'a> ScopeDisposer<'a> {
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
    pub fn leak(self) -> Scope<'a> {
        let cx = self.0;
        std::mem::forget(self);
        cx
    }

    /// Constructs a [`ScopeDisposer`] from the leaked [`Scope`].
    ///
    /// # Safety
    ///
    /// This function is unsafe because a [`Scope`] may be disposed twice and
    /// lead an undefined behavior.
    pub unsafe fn from_leaked(cx: Scope<'a>) -> Self {
        Self(cx)
    }
}

unsafe fn free<T: ?Sized>(mut ptr: NonNull<T>) {
    std::ptr::drop_in_place(ptr.as_mut());
}

impl Drop for ScopeDisposer<'_> {
    fn drop(&mut self) {
        SHARED.with(|shared| {
            let Scope { id, .. } = self.0;
            // 1) Cleanup resources created inside this `Scope`.
            let cleanups = id.with(shared, |cx| std::mem::take(&mut cx.cleanups));
            for cl in cleanups.into_iter().rev() {
                // SAFETY: It's safe to destroy allocated values because `Signal`s
                // `Effect`s and `Variable`s need to their IDs to request the `Shared`
                // object and check the accessibility of owned resources and once these
                // allocated resources are disposed their IDs are no longer available.
                #[allow(clippy::undocumented_unsafe_blocks)]
                match cl {
                    Cleanup::Signal(id) => {
                        let ptr = shared
                            .signals
                            .borrow_mut()
                            .remove(id)
                            .unwrap_or_else(|| unreachable!());
                        unsafe { free(ptr) };
                        shared.signal_contexts.borrow_mut().remove(id);
                    }
                    Cleanup::Effect(id) => {
                        let ptr = shared
                            .effects
                            .borrow_mut()
                            .remove(id)
                            .unwrap_or_else(|| unreachable!());
                        unsafe { free(ptr) };
                        shared.effect_contexts.borrow_mut().remove(id);
                    }
                    Cleanup::Variable(id) => {
                        let ptr = shared
                            .variables
                            .borrow_mut()
                            .remove(id)
                            .unwrap_or_else(|| unreachable!());
                        unsafe { free(ptr) };
                    }
                    Cleanup::Callback(mut cb) => unsafe {
                        cb.as_mut().call_once(shared);
                        free(cb);
                    },
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
    unsafe fn alloc<'a, T: 'a>(&self, t: T) -> &'a T {
        std::mem::transmute(self.arena.alloc(t))
    }

    pub unsafe fn alloc_var<'a, T: 'a>(&self, t: T) -> &'a VarSlot<T> {
        self.alloc(VarSlot::new(t))
    }

    pub fn add_cleanup(&mut self, cl: Cleanup) {
        self.cleanups.push(cl);
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

pub fn create_root<'disposer>(
    f: impl for<'root> FnOnce(BoundedScope<'root, 'disposer>),
) -> ScopeDisposer<'disposer> {
    let disposer = ScopeDisposer::new(None);
    f(disposer.0);
    disposer
}

impl<'a> Scope<'a> {
    pub(crate) fn with_shared<T>(&self, f: impl FnOnce(&'a Shared) -> T) -> T {
        // SAFETY: This is safe because the reference can't escape from the closure.
        SHARED.with(|shared| f(unsafe { std::mem::transmute(shared) }))
    }

    pub fn create_child(
        &self,
        f: impl for<'child> FnOnce(BoundedScope<'child, 'a>),
    ) -> ScopeDisposer<'a> {
        let disposer = ScopeDisposer::new(Some(self.id));
        // SAFETY: Since the 'child lifetime is shorter than 'a and child scope
        // can't escape from the closure, we can safely transmute the lifetime.
        f(unsafe { std::mem::transmute(disposer.0) });
        disposer
    }

    /// Allocated a new arbitrary value under current scope.
    ///
    /// # Safety
    ///
    /// This function is unsafe because the given value may hold a reference
    /// to another variable allocated after it. If the value read that reference
    /// while being disposed, this will result in an undefined behavior.
    ///
    /// For using this function safely, you must assume the specified value
    /// will not read any references created after itself, and will not be read
    /// by any references created before itself.
    pub unsafe fn alloc<T: 'a>(&self, t: T) -> &'a T {
        self.with_shared(|shared| {
            self.id.with(shared, |cx| {
                let ptr = cx.alloc(t);
                let any_ptr = std::mem::transmute(NonNull::from(ptr as &dyn Empty));
                let id = shared.variables.borrow_mut().insert(any_ptr);
                cx.add_cleanup(Cleanup::Variable(id));
                ptr
            })
        })
    }

    pub fn create_cell<T: 'a + Copy>(&self, t: T) -> &'a Cell<T> {
        // SAFETY: A `Copy` type can't impelemt `Drop`, therefore the underlying
        // value can't be read during disposal.
        unsafe { self.alloc(Cell::new(t)) }
    }

    pub fn untrack(&self, f: impl FnOnce()) {
        self.with_shared(|shared| shared.untrack(f));
    }

    pub fn on_cleanup(&self, f: impl FnOnce()) {
        self.with_shared(|shared| {
            self.id.with(shared, |cx| {
                // SAFETY: Same as creating variables.
                let cb = unsafe {
                    let ptr = cx.alloc(f);
                    std::mem::transmute(NonNull::from(ptr as &dyn Callback))
                };
                cx.add_cleanup(Cleanup::Callback(cb));
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleanup() {
        create_root(|cx| {
            let called = cx.create_signal(0);
            cx.create_child(|cx| {
                cx.on_cleanup(|| {
                    called.update(|x| *x + 1);
                });
            });
            assert_eq!(*called.get(), 1);
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
                counter.update(|x| *x + 1);

                cx.on_cleanup(|| {
                    trigger_cleanup.track();
                });
            });
            assert_eq!(*counter.get(), 1);

            // Trigger effect to let the scope be disposed and run the cleanup.
            trigger_effect.trigger();
            assert_eq!(*counter.get(), 2);

            // This should not work since the cleanup is untracked.
            trigger_cleanup.trigger();
            assert_eq!(*counter.get(), 2);
        });
    }

    #[test]
    fn cleanup_in_scoped_effect() {
        create_root(|cx| {
            let trigger = cx.create_signal(());
            let counter = cx.create_signal(0);

            cx.create_effect_scoped(move |cx| {
                trigger.track();
                cx.on_cleanup(|| {
                    counter.update(|x| *x + 1);
                });
            });
            assert_eq!(*counter.get(), 0);

            trigger.trigger();
            assert_eq!(*counter.get(), 1);

            // A new scope is disposed.
            trigger.trigger();
            assert_eq!(*counter.get(), 2);
        });
    }
}
