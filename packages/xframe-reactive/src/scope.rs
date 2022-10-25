use crate::{
    arena::{Arena, GBump, WeakRef},
    context::Contexts,
    effect::RawEffect,
    signal::SignalContext,
};
use std::{cell::Cell, fmt, marker::PhantomData, ops::Deref};

pub type Scope<'a> = BoundedScope<'a, 'a>;

#[derive(Clone, Copy)]
pub struct BoundedScope<'a, 'b: 'a> {
    inner: &'a ScopeInner<'a>,
    /// The 'b life bounds is requred because we need to tell the compiler
    /// the child scope should never outlives its parent.
    bounds: PhantomData<&'b ()>,
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

    pub(crate) fn shared(&self) -> BoundedScopeShared<'a> {
        self.inner.inherited.shared
    }
}

struct ScopeInner<'a> {
    arena: Arena<'a>,
    inherited: ScopeInherited<'a>,
    // Ensure the 'a lifebounds is invariance.
    phantom: PhantomData<&'a mut &'a mut ()>,
}

impl fmt::Debug for ScopeInner<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scope")
            .field("arena", &self.arena)
            .field("inherited", &self.inherited)
            .finish()
    }
}

pub(crate) struct ScopeInherited<'a> {
    pub parent: Option<&'a ScopeInherited<'a>>,
    pub contexts: Contexts<'a>,
    shared: BoundedScopeShared<'a>,
}

impl fmt::Debug for ScopeInherited<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScopeInherited")
            .field("parent", &self.parent.map(|x| x as *const ScopeInherited))
            .field("contexts", &self.contexts)
            .field("shared", &(self.shared.inner as *const ScopeShared))
            .finish()
    }
}

pub(crate) type EffectRef = WeakRef<'static, RawEffect<'static>>;
pub(crate) type SignalContextRef = WeakRef<'static, SignalContext>;

#[derive(Clone, Copy)]
pub(crate) struct BoundedScopeShared<'a> {
    inner: &'static ScopeShared,
    bounds: PhantomData<&'a ()>,
}

impl Deref for BoundedScopeShared<'_> {
    type Target = &'static ScopeShared;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[derive(Default)]
pub(crate) struct ScopeShared {
    pub observer: Cell<Option<EffectRef>>,
    pub singal_contexts: GBump<SignalContext>,
    pub effects: GBump<RawEffect<'static>>,
}

pub struct ScopeDisposer<'a> {
    scope: Option<Box<ScopeInner<'a>>>,
}

impl fmt::Debug for ScopeDisposer<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ScopeDisposer").field(&self.scope).finish()
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
                let shared = {
                    let boxed = Box::new(ScopeShared::default());
                    BoundedScopeShared {
                        inner: Box::leak(boxed),
                        bounds: PhantomData,
                    }
                };
                ScopeInherited {
                    parent: None,
                    contexts: Default::default(),
                    shared,
                }
            });
        let scope = Box::new(ScopeInner {
            arena: Default::default(),
            inherited,
            phantom: PhantomData,
        });
        ScopeDisposer { scope: Some(scope) }
    }

    fn new_within(
        parent: Option<&'a ScopeInherited<'a>>,
        f: impl for<'b> FnOnce(BoundedScope<'b, 'a>),
    ) -> Self {
        let scope = ScopeDisposer::new(parent).leak();
        f(scope);
        // SAFETY: no variables escape from the closure `f`, it's safe to
        // dispose the scope.
        unsafe { ScopeDisposer::from_leaked(scope) }
    }

    pub fn leak(mut self) -> Scope<'a> {
        self.scope
            .take()
            .map(Box::leak)
            .map(|inner| BoundedScope {
                inner,
                bounds: PhantomData,
            })
            .unwrap_or_else(|| unreachable!())
    }

    /// # Safety
    ///
    /// This function is unsafe because a scope might be disposed twice, and
    /// there may be references to variables created in the scope.
    pub unsafe fn from_leaked(scope: Scope<'a>) -> Self {
        let scope = Box::from_raw(scope.inner as *const ScopeInner as *mut ScopeInner);
        ScopeDisposer { scope: Some(scope) }
    }
}

impl Drop for ScopeDisposer<'_> {
    fn drop(&mut self) {
        if let Some(scope) = &self.scope {
            unsafe { scope.arena.dispose() };
            if scope.inherited.parent.is_none() {
                let shared = unsafe {
                    Box::from_raw(
                        scope.inherited.shared.inner as *const ScopeShared as *mut ScopeShared,
                    )
                };
                drop(shared);
            }
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
        ScopeDisposer::new_within(Some(&self.inner.inherited), f)
    }

    pub fn create_variable<T: 'a>(self, t: T) -> &'a T {
        self.inner.arena.alloc(t)
    }

    pub fn untrack(self, f: impl FnOnce()) {
        let sub = &self.inner.inherited.shared.observer;
        let saved = self.inner.inherited.shared.observer.take();
        f();
        sub.set(saved);
    }
}
