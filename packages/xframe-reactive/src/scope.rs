use crate::{
    arena::{Arena, WeakRef},
    context::Contexts,
    effect::RawEffect,
    signal::RawSignal,
    CovariantLifetime, InvariantLifetime,
};
use bumpalo::Bump;
use smallvec::SmallVec;
use std::{
    cell::{Cell, RefCell},
    fmt,
    marker::PhantomData,
};

const INITIAL_CLEANUP_SLOTS: usize = 4;

pub type Scope<'a> = BoundedScope<'a, 'a>;

#[derive(Clone, Copy)]
pub struct BoundedScope<'a, 'b: 'a> {
    inner: &'a RawScope<'a>,
    /// The 'b life bounds is requred because we need to tell the compiler
    /// the child scope should never outlives its parent.
    bounds: PhantomData<(InvariantLifetime<'a>, CovariantLifetime<'b>)>,
}

impl fmt::Debug for Scope<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(f)
    }
}

impl<'a> Scope<'a> {
    pub(crate) fn inherited(&self) -> &'a ScopeInherited<'a> {
        &self.inner.inherited
    }

    pub(crate) fn shared(&self) -> &'static Shared {
        self.inner.inherited.shared
    }

    pub(crate) fn push_cleanup(&self, ty: Cleanup<'a>) {
        self.inner.cleanups.borrow_mut().push(ty);
    }
}

trait Empty {}
impl<T> Empty for T {}

pub(crate) struct Variable<'a>(&'a dyn Empty);

pub(crate) enum Cleanup<'a> {
    Signal(SignalRef),
    Effect(EffectRef),
    Variable(Variable<'a>),
    Callback(Box<dyn 'a + FnOnce()>),
}

struct RawScope<'a> {
    variables: Bump,
    inherited: ScopeInherited<'a>,
    cleanups: RefCell<SmallVec<[Cleanup<'a>; INITIAL_CLEANUP_SLOTS]>>,
}

impl fmt::Debug for RawScope<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scope")
            .field("arena", &self.variables)
            .field("inherited", &self.inherited)
            .finish()
    }
}

impl Drop for RawScope<'_> {
    fn drop(&mut self) {
        // SAFETY: last alloced variables must be disposed first because signals
        // and effects need to do some cleanup works with its captured references.
        for ty in self.cleanups.take().into_iter().rev() {
            match ty {
                Cleanup::Signal(sig) => {
                    self.inherited.shared.signals.free(sig);
                }
                Cleanup::Effect(eff) => {
                    self.inherited.shared.effects.free(eff);
                }
                Cleanup::Variable(ptr) => unsafe {
                    std::ptr::drop_in_place(ptr.0 as *const dyn Empty as *mut dyn Empty);
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
            .field("shared", &(self.shared as *const Shared))
            .finish()
    }
}

pub(crate) type SignalRef = WeakRef<'static, RawSignal<'static>>;
pub(crate) type EffectRef = WeakRef<'static, RawEffect<'static>>;

#[derive(Default)]
pub(crate) struct Shared {
    pub observer: Cell<Option<WeakRef<'static, RawEffect<'static>>>>,
    pub signals: Arena<RawSignal<'static>>,
    pub effects: Arena<RawEffect<'static>>,
    scopes: Arena<RawScope<'static>>,
}

pub struct ScopeDisposer<'a> {
    scope: WeakRef<'static, RawScope<'static>>,
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
        let raw = RawScope {
            variables: Default::default(),
            inherited,
            cleanups: Default::default(),
        };
        let raw = unsafe { std::mem::transmute(raw) };
        let scope = shared.scopes.alloc(raw);
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
    pub unsafe fn leak_scope(&self) -> Scope<'a> {
        BoundedScope {
            inner: std::mem::transmute(self.scope.leak_ref()),
            bounds: PhantomData,
        }
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

impl<'a> Scope<'a> {
    pub fn create_root(f: impl for<'b> FnOnce(BoundedScope<'b, 'a>)) -> ScopeDisposer<'a> {
        ScopeDisposer::new_within(None, f)
    }

    pub fn create_child(
        self,
        f: impl for<'child> FnOnce(BoundedScope<'child, 'a>),
    ) -> ScopeDisposer<'a> {
        ScopeDisposer::new_within(Some(&self.inherited()), f)
    }

    pub fn create_variable<T: 'a>(self, t: T) -> &'a T {
        let ptr = &*self.inner.variables.alloc(t);
        self.push_cleanup(Cleanup::Variable(Variable(ptr)));
        ptr
    }

    pub fn on_cleanup(&self, f: impl 'a + FnOnce()) {
        self.push_cleanup(Cleanup::Callback(Box::new(f)));
    }

    pub fn untrack(self, f: impl FnOnce()) {
        let observer = &self.shared().observer;
        let saved = observer.take();
        f();
        observer.set(saved);
    }
}
