use crate::{
    arena::{Arena, WeakRef},
    effect::OwnedEffect,
    scope::BoundedOwnedScope,
    signal::SignalContext,
};
use std::{cell::Cell, fmt};

pub(crate) type InvariantLifetime<'a> = &'a mut &'a mut ();
pub(crate) type CovariantLifetime<'a> = &'a ();

pub(crate) type SignalContextRef = WeakRef<'static, SignalContext>;
pub(crate) type EffectRef = WeakRef<'static, OwnedEffect<'static>>;

pub(crate) trait Empty {}
impl<T> Empty for T {}

#[derive(Default)]
pub(crate) struct Shared {
    pub observer: Cell<Option<WeakRef<'static, OwnedEffect<'static>>>>,
    pub signal_contexts: Arena<SignalContext>,
    pub effects: Arena<OwnedEffect<'static>>,
    pub scopes: Arena<BoundedOwnedScope<'static, 'static>>,
}

impl fmt::Debug for Shared {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Shared")
            .field("observer", &self.observer.get().map(|x| x.as_ptr()))
            .finish_non_exhaustive()
    }
}
