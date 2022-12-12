use crate::{
    context::ScopeContexts,
    effect::{AnyEffect, EffectContext},
    scope::RawScope,
    signal::SignalContext,
};
use slotmap::{new_key_type, SecondaryMap, SlotMap, SparseSecondaryMap};
use std::{
    any::Any,
    cell::{Cell, RefCell},
    rc::Rc,
};

thread_local! {
    pub(crate) static SHARED: Shared = <_>::default();
}

new_key_type! {
    pub(crate) struct ScopeId;
    pub(crate) struct SignalId;
    pub(crate) struct EffectId;
    pub(crate) struct VariableId;
}

type ASparseSecondaryMap<K, V> = SparseSecondaryMap<K, V, ahash::RandomState>;

#[derive(Default)]
pub(crate) struct Shared {
    pub observer: Cell<Option<EffectId>>,
    pub scopes: RefCell<SlotMap<ScopeId, RawScope>>,
    pub scope_parents: RefCell<SecondaryMap<ScopeId, ScopeId>>,
    pub scope_contexts: RefCell<ASparseSecondaryMap<ScopeId, ScopeContexts>>,
    pub signals: RefCell<SlotMap<SignalId, Box<dyn Any>>>,
    pub signal_contexts: RefCell<SecondaryMap<SignalId, SignalContext>>,
    pub effects: RefCell<SlotMap<EffectId, Rc<RefCell<dyn AnyEffect>>>>,
    pub effect_contexts: RefCell<SecondaryMap<EffectId, EffectContext>>,
    pub variables: RefCell<SlotMap<VariableId, Box<dyn Any>>>,
}
