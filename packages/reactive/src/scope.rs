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
    callbacks: RefCell<Vec<&'a mut (dyn 'a + Callback)>>,
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
        // Last allocated variables will be disposed first.
        for cl in self.cleanups.borrow().iter().copied().rev() {
            match cl {
                Cleanup::SignalContext(sig) => {
                    self.inherited.shared.signal_contexts.free(sig);
                }
                Cleanup::Effect(eff) => {
                    self.inherited.shared.effects.free(eff);
                }
                Cleanup::Variable(ptr) => unsafe {
                    std::ptr::drop_in_place(ptr as *const dyn Empty as *mut dyn Empty);
                },
            }
        }
        // Callbacks should be invoked at last, since they can read a variable
        // allocated after themselves.
        for cb in self.callbacks.take().into_iter().rev() {
            unsafe { cb.call_once() };
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

/// Helper trait to invoke [`FnOnce`].
///
/// Original post: <https://users.rust-lang.org/t/invoke-mut-dyn-fnonce/59356/4>
pub(crate) trait Callback {
    /// Please make sure this will only be called once!
    unsafe fn call_once(&mut self);
}

impl<F: FnOnce()> Callback for F {
    unsafe fn call_once(&mut self) {
        std::ptr::read(self)()
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Cleanup<'a> {
    SignalContext(SignalContextRef),
    Effect(EffectRef),
    Variable(&'a (dyn 'a + Empty)),
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
            callbacks: Default::default(),
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

    /// Leak a bounded scope reference. Sometimes it's useful to use a scope
    /// outside a closure.
    ///
    /// # Safety
    ///
    /// This function is unsafe, because:
    ///
    /// 1. There may be references to variables allocated by the returned scope
    /// when the disposer is dropped, read those references will cause undefined
    /// behavior;
    /// 2. If a `Scope<'static>` is leaked from the disposer, use that scope will
    /// break the safety guarantee of [`create_variable_static`](OwnedScope::create_variable_static).
    pub unsafe fn leak_scope<'b>(&'b self) -> BoundedScope<'b, 'a> {
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

    /// Allocated a new arbitrary value under current scope.
    ///
    /// # Safety
    ///
    /// This function is unsafe because the given value may hold a reference
    /// to another variable allocated after it. If the value read that reference
    /// while being disposed, this will result in an undefined behavior.
    ///
    /// For using this function safely, you must ensure the specified value
    /// will not read any references created after itself.
    ///
    /// Check out <https://github.com/loichyan/xframe/issues/1> for more details.
    pub unsafe fn create_variable_unchecked<T>(&'a self, t: T) -> &'a T {
        let ptr = &*self.variables.alloc(t);
        if std::mem::needs_drop::<T>() {
            self.push_cleanup(Cleanup::Variable(ptr));
        }
        ptr
    }

    pub fn create_variable_copied<T: Copy>(&'a self, t: T) -> &'a T {
        // SAFETY: A `Copy` type can't have a custom destructor.
        unsafe { self.create_variable_unchecked(t) }
    }

    pub fn create_variable_static<T: 'static>(&'a self, t: T) -> &'a T {
        // SAFETY: A 'static type can't hold a 'a reference.
        unsafe { self.create_variable_unchecked(t) }
    }

    pub fn on_cleanup(&'a self, f: impl 'a + FnOnce()) {
        let f = self.variables.alloc(|| self.untrack(f));
        self.callbacks.borrow_mut().push(f);
    }

    pub fn untrack(&'a self, f: impl FnOnce()) {
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
            let var = cx.create_variable_static(1);
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
            cx.create_variable_static(DropAndInc);
            cx.create_child(|cx| {
                cx.create_variable_static(DropAndAssert(1));
                cx.create_variable_static(DropAndInc);
                cx.create_variable_static(DropAndAssert(0));
            });
        });
        drop(DropAndAssert(2));
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
            let var = cx.create_variable_static(777);
            unsafe { cx.create_variable_unchecked(AssertVarOnDrop { var, expect: 777 }) };
        });
    }
}
