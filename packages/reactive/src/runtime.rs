use crate::{
    context::ScopeContexts,
    effect::{EffectContext, RawEffect},
    scope::RawScope,
    signal::{RawSignal, SignalContext},
};
use slotmap::{new_key_type, SecondaryMap, SlotMap, SparseSecondaryMap};
use std::cell::{Cell, RefCell};

thread_local! {
    pub(crate) static RT: Runtime = <_>::default();
}

new_key_type! {
    pub(crate) struct ScopeId;
    pub(crate) struct SignalId;
    pub(crate) struct EffectId;
}

type ASparseSecondaryMap<K, V> = SparseSecondaryMap<K, V, ahash::RandomState>;

#[derive(Default)]
pub(crate) struct Runtime {
    pub observer: Cell<Option<EffectId>>,
    pub scopes: RefCell<SlotMap<ScopeId, RawScope>>,
    pub scope_parents: RefCell<SecondaryMap<ScopeId, ScopeId>>,
    pub scope_contexts: RefCell<ASparseSecondaryMap<ScopeId, ScopeContexts>>,
    pub signals: RefCell<SlotMap<SignalId, RawSignal>>,
    pub signal_contexts: RefCell<SecondaryMap<SignalId, SignalContext>>,
    pub effects: RefCell<SlotMap<EffectId, RawEffect>>,
    pub effect_contexts: RefCell<SecondaryMap<EffectId, EffectContext>>,
}
