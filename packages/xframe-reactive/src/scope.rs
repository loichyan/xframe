use crate::{
    arena::{Arena, Disposer},
    context::Contexts,
    effect::RawEffect,
};
use std::{
    cell::{Cell, RefCell},
    marker::PhantomData,
};

pub type Scope<'a> = BoundedScope<'a, 'a>;

#[derive(Debug, Clone, Copy)]
pub struct BoundedScope<'a, 'b: 'a> {
    inner: &'a ScopeInner<'a>,
    /// The 'b life bounds is requred because we need to tell the compiler
    /// the child scope should never outlives its parent.
    bounds: PhantomData<&'b ()>,
}

impl<'a> Scope<'a> {
    pub(crate) fn shared(&self) -> &'static ScopeShared {
        self.inner.inherited.shared
    }

    pub(crate) fn alloc_effect(&self, raw: RawEffect<'a>) -> &'a RawEffect<'a> {
        let (eff, disposer) = self.inner.arena.alloc(raw);
        self.inner.effects.borrow_mut().push(disposer);
        eff
    }

    pub(crate) fn inherited(&self) -> &'a ScopeInherited<'a> {
        &self.inner.inherited
    }
}

#[derive(Debug)]
struct ScopeInner<'a> {
    arena: Arena,
    inherited: ScopeInherited<'a>,
    effects: RefCell<Vec<Disposer>>,
    variables: RefCell<Vec<Disposer>>,
}

#[derive(Debug)]
pub(crate) struct ScopeInherited<'a> {
    pub parent: Option<&'a ScopeInherited<'a>>,
    pub contexts: Contexts<'a>,
    shared: &'static ScopeShared,
}

#[derive(Debug, Default)]
pub(crate) struct ScopeShared {
    pub subscriber: Cell<Option<&'static RawEffect<'static>>>,
}

#[derive(Debug)]
pub struct ScopeDisposer<'a> {
    inner: &'a ScopeInner<'a>,
    is_root: bool,
}

impl Drop for ScopeDisposer<'_> {
    fn drop(&mut self) {
        let mut inner =
            unsafe { Box::from_raw(self.inner as *const ScopeInner as *mut ScopeInner) };
        for eff in inner.effects.get_mut() {
            unsafe {
                eff.dispose();
            }
        }
        for var in inner.variables.get_mut() {
            unsafe {
                var.dispose();
            }
        }
        if self.is_root {
            let shared = unsafe {
                Box::from_raw(inner.inherited.shared as *const ScopeShared as *mut ScopeShared)
            };
            drop(shared);
        }
    }
}

fn create_scope_inner<'a>(parent: Option<&'a ScopeInherited<'a>>) -> &'a ScopeInner<'a> {
    let inherited = parent
        .map(|parent| ScopeInherited {
            parent: Some(parent),
            contexts: Default::default(),
            shared: parent.shared,
        })
        .unwrap_or_else(|| {
            let shared = Box::new(ScopeShared::default());

            ScopeInherited {
                parent: None,
                contexts: Default::default(),
                shared: Box::leak(shared),
            }
        });
    let boxed = Box::new(ScopeInner {
        arena: Default::default(),
        inherited,
        effects: Default::default(),
        variables: Default::default(),
    });
    &*Box::leak(boxed)
}

impl<'a> Scope<'a> {
    pub fn create_root<'disposer>(f: impl for<'b> FnOnce(Scope<'b>)) -> ScopeDisposer<'disposer> {
        let inner = create_scope_inner(None);
        f(Scope {
            inner,
            bounds: PhantomData,
        });
        ScopeDisposer {
            inner,
            is_root: true,
        }
    }

    pub fn create_child(
        self,
        f: impl for<'child> FnOnce(BoundedScope<'child, 'a>),
    ) -> ScopeDisposer<'a> {
        let inner = create_scope_inner(Some(&self.inner.inherited));
        f(Scope {
            inner,
            bounds: PhantomData,
        });
        ScopeDisposer {
            inner,
            is_root: false,
        }
    }

    pub fn create_variable<T: 'a>(self, t: T) -> &'a T {
        let (val, disposer) = self.inner.arena.alloc(t);
        self.inner.variables.borrow_mut().push(disposer);
        val
    }

    pub fn untrack(self, f: impl FnOnce()) {
        let sub = &self.inner.inherited.shared.subscriber;
        let saved = self.inner.inherited.shared.subscriber.take();
        f();
        sub.set(saved);
    }
}
