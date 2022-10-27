use crate::{
    arena::WeakRef,
    context::Contexts,
    shared::{CovariantLifetime, EffectRef, Empty, InvariantLifetime, Shared, SignalContextRef},
};
use bumpalo::Bump;
use smallvec::SmallVec;
use std::{cell::RefCell, fmt, marker::PhantomData, ops::Deref};

const INITIAL_CLEANUP_SLOTS: usize = 4;

pub type Scope<'a> = &'a OwnedScope<'a>;
pub type BoundedScope<'a, 'b> = &'a BoundedOwnedScope<'a, 'b>;

pub struct BoundedOwnedScope<'a, 'b: 'a> {
    scope: OwnedScope<'a>,
    /// The 'b life bounds is requred because we need to tell the compiler
    /// the child scope should never outlives its parent.
    bounds: PhantomData<(InvariantLifetime<'a>, CovariantLifetime<'b>)>,
}

impl<'a> Deref for BoundedOwnedScope<'a, '_> {
    type Target = OwnedScope<'a>;

    fn deref(&self) -> &Self::Target {
        &self.scope
    }
}

impl fmt::Debug for BoundedOwnedScope<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.scope.fmt(f)
    }
}

pub struct OwnedScope<'a> {
    variables: Bump,
    inherited: ScopeInherited<'a>,
    cleanups: RefCell<SmallVec<[Cleanup<'a>; INITIAL_CLEANUP_SLOTS]>>,
}

impl fmt::Debug for OwnedScope<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scope")
            .field("inherited", &self.inherited)
            .field("cleanups", &*self.cleanups.borrow() as _)
            .finish_non_exhaustive()
    }
}

impl Drop for OwnedScope<'_> {
    fn drop(&mut self) {
        // SAFETY: last alloced variables must be disposed first because signals
        // and effects need to do some cleanup works with its captured references.
        for ty in self.cleanups.take().into_iter().rev() {
            match ty {
                Cleanup::SignalContext(sig) => {
                    self.inherited.shared.signal_contexts.free(sig);
                }
                Cleanup::Effect(eff) => {
                    self.inherited.shared.effects.free(eff);
                }
                Cleanup::Variable(ptr) => unsafe {
                    std::ptr::drop_in_place(ptr as *const dyn Empty as *mut dyn Empty);
                },
                Cleanup::Callback(f) => f(),
            }
        }
    }
}

pub(crate) struct ScopeInherited<'a> {
    pub parent: Option<&'a ScopeInherited<'a>>,
    pub contexts: Contexts<'a>,
    shared: &'static Shared,
}

impl fmt::Debug for ScopeInherited<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScopeInherited")
            .field("parent", &self.parent.map(|x| x as *const ScopeInherited))
            .field("contexts", &self.contexts)
            .field("shared", &self.shared)
            .finish()
    }
}

pub(crate) enum Cleanup<'a> {
    SignalContext(SignalContextRef),
    Effect(EffectRef),
    Variable(&'a (dyn 'a + Empty)),
    // TODO: can we use a bumpalo::boxed::Box here?
    Callback(Box<dyn 'a + FnOnce()>),
}

impl fmt::Debug for Cleanup<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Cleanup::SignalContext(s) => f.debug_tuple("Signal").field(&s.as_ptr()).finish(),
            Cleanup::Effect(e) => f.debug_tuple("Effect").field(&e.as_ptr()).finish(),
            Cleanup::Variable(v) => f
                .debug_tuple("Variable")
                .field(&(v as *const dyn Empty))
                .finish(),
            Cleanup::Callback(c) => f
                .debug_tuple("Callback")
                .field(&(&*c as *const dyn FnOnce()))
                .finish(),
        }
    }
}

pub struct ScopeDisposer<'a> {
    scope: WeakRef<'static, BoundedOwnedScope<'static, 'static>>,
    bounds: PhantomData<InvariantLifetime<'a>>,
}

impl fmt::Debug for ScopeDisposer<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ScopeDisposer")
            .field(&self.scope.as_ptr())
            .finish()
    }
}

impl<'a> ScopeDisposer<'a> {
    fn new(parent: Option<&'a ScopeInherited<'a>>) -> Self {
        let inherited = parent
            .map(|parent| ScopeInherited {
                parent: Some(parent),
                contexts: Default::default(),
                shared: parent.shared,
            })
            .unwrap_or_else(|| {
                let shared = Box::new(Shared::default());
                ScopeInherited {
                    parent: None,
                    contexts: Default::default(),
                    shared: Box::leak(shared),
                }
            });
        let shared = inherited.shared;
        let scope = OwnedScope {
            variables: Default::default(),
            inherited,
            cleanups: Default::default(),
        };
        let bounded = BoundedOwnedScope {
            scope,
            bounds: PhantomData,
        };
        let bounded = unsafe { std::mem::transmute(bounded) };
        let scope = shared.scopes.alloc(bounded);
        ScopeDisposer {
            scope,
            bounds: PhantomData,
        }
    }

    fn new_within(
        parent: Option<&'a ScopeInherited<'a>>,
        f: impl for<'b> FnOnce(BoundedScope<'b, 'a>),
    ) -> Self {
        let disposer = ScopeDisposer::new(parent);
        // SAFETY: no variables escape from the closure `f`, it's safe to
        // dispose the scope.
        unsafe { f(disposer.leak_scope()) };
        disposer
    }

    /// # Safety
    ///
    /// This function is unsafe because a scope might be disposed twice, and
    /// there may be references to variables created in the scope.
    pub unsafe fn leak_scope(&self) -> BoundedScope<'a, 'a> {
        std::mem::transmute(self.scope.leak_ref())
    }
}

impl Drop for ScopeDisposer<'_> {
    fn drop(&mut self) {
        let (shared, is_root) = self
            .scope
            .with(|scope| (scope.inherited.shared, scope.inherited.parent.is_none()))
            .unwrap_or_else(|| unreachable!());
        shared.scopes.free(self.scope);
        if is_root {
            let shared = unsafe { Box::from_raw(shared as *const Shared as *mut Shared) };
            drop(shared);
        }
    }
}

pub fn create_root<'a>(f: impl for<'b> FnOnce(BoundedScope<'b, 'a>)) -> ScopeDisposer<'a> {
    ScopeDisposer::new_within(None, f)
}

impl<'a> OwnedScope<'a> {
    pub(crate) fn inherited(&self) -> &ScopeInherited<'a> {
        &self.inherited
    }

    pub(crate) fn shared(&self) -> &'static Shared {
        self.inherited.shared
    }

    pub(crate) fn push_cleanup(&self, ty: Cleanup<'a>) {
        self.cleanups.borrow_mut().push(ty);
    }

    pub fn create_child(
        &'a self,
        f: impl for<'child> FnOnce(BoundedScope<'child, 'a>),
    ) -> ScopeDisposer<'a> {
        ScopeDisposer::new_within(Some(&self.inherited()), f)
    }

    pub fn create_variable<T: 'a>(&'a self, t: T) -> &'a T {
        let ptr = &*self.variables.alloc(t);
        self.push_cleanup(Cleanup::Variable(ptr));
        ptr
    }

    pub fn on_cleanup(&'a self, f: impl 'a + FnOnce()) {
        self.push_cleanup(Cleanup::Callback(Box::new(|| self.untrack(f))));
    }

    pub fn untrack(&self, f: impl FnOnce()) {
        let observer = &self.shared().observer;
        let saved = observer.take();
        f();
        observer.set(saved);
    }
}

#[cfg(test)]
mod test {
    use super::create_root;
    use std::cell::Cell;

    #[test]
    fn variables() {
        let a = Cell::new(-1);
        create_root(|cx| {
            let var = cx.create_variable(1);
            a.set(*var);
        });
        assert_eq!(a.get(), 1);
    }

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
    fn cleanup_in_scoped_effect() {
        create_root(|cx| {
            let trigger = cx.create_signal(());
            let counter = cx.create_signal(0);

            cx.create_effect_scoped(|cx| {
                trigger.track();
                cx.on_cleanup(|| {
                    counter.update(|x| *x + 1);
                });
            });
            assert_eq!(*counter.get(), 0);

            trigger.trigger_subscribers();
            assert_eq!(*counter.get(), 1);

            trigger.trigger_subscribers();
            assert_eq!(*counter.get(), 2);
        });
    }

    #[test]
    fn cleanup_is_untracked() {
        create_root(|cx| {
            let trigger = cx.create_signal(());
            let trigger2 = cx.create_signal(());
            let counter = cx.create_signal(0);

            cx.create_effect_scoped(|cx| {
                trigger.track();
                counter.update(|x| *x + 1);

                cx.on_cleanup(|| {
                    trigger2.track();
                });
            });
            assert_eq!(*counter.get(), 1);

            trigger.trigger_subscribers(); // Call the cleanup
            assert_eq!(*counter.get(), 2);

            trigger2.trigger_subscribers();
            assert_eq!(*counter.get(), 2);
        });
    }

    #[test]
    fn store_disposer_in_own_signal() {
        create_root(|cx| {
            let state = cx.create_signal(None);
            let disposer = cx.create_child(|_| ());
            state.set(Some(disposer));
        });
    }

    #[test]
    fn drop_variables_on_dispose() {
        thread_local! {
            static COUNTER: Cell<i32> = Cell::new(0);
        }

        struct DropAndInc;
        impl Drop for DropAndInc {
            fn drop(&mut self) {
                COUNTER.with(|x| x.set(x.get() + 1));
            }
        }

        struct DropAndAssert(i32);
        impl Drop for DropAndAssert {
            fn drop(&mut self) {
                assert_eq!(COUNTER.with(Cell::get), self.0);
            }
        }

        create_root(|cx| {
            cx.create_variable(DropAndInc);
            cx.create_child(|cx| {
                cx.create_variable(DropAndAssert(1));
                cx.create_variable(DropAndInc);
                cx.create_variable(DropAndAssert(0));
            });
        });
        assert_eq!(COUNTER.with(Cell::get), 2);
    }

    #[test]
    fn access_previous_var_on_drop() {
        struct AssertVarOnDrop<'a> {
            var: &'a i32,
            expect: i32,
        }
        impl Drop for AssertVarOnDrop<'_> {
            fn drop(&mut self) {
                assert_eq!(*self.var, self.expect);
            }
        }

        create_root(|cx| {
            let var = cx.create_variable(777);
            cx.create_variable(AssertVarOnDrop { var, expect: 777 });
        });
    }
}
