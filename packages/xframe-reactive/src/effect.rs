use crate::{
    scope::{Scope, ScopeShared},
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

        // Re-calculate dependencies.
        self.clear_dependencies();

        // Save previous subscriber.
        let saved = self.shared.0.subscriber.take();
        self.shared.0.subscriber.set(Some(this));

        // Call the effect.
        self.effect.run();

        // Notify all captured signals subscribe itself.
        for dep in self.dependencies.borrow().iter() {
            dep.0.subscribe(this);
        }

        // Restore previous subscriber.
        self.shared.0.subscriber.set(saved);
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
    let inner = cx.alloc_effect(RawEffect {
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
}
