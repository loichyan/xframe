use crate::{
    context::ScopeContexts,
    effect::{EffectContext, RawEffect},
    scope::RawScope,
    signal::{RawSignal, SignalContext},
};
use slotmap::{new_key_type, SecondaryMap, SlotMap};
use std::cell::{Cell, RefCell};

#[cfg(feature = "fxhash")]
pub(crate) type RandomState = std::hash::BuildHasherDefault<rustc_hash::FxHasher>;
#[cfg(not(feature = "fxhash"))]
pub(crate) type RandomState = std::collections::hash_map::RandomState;

pub(crate) type HashMap<K, V> = std::collections::hash_map::HashMap<K, V, RandomState>;
pub(crate) type HashSet<T> = std::collections::hash_set::HashSet<T, RandomState>;
pub(crate) type IndexSet<T> = indexmap::IndexSet<T, RandomState>;
pub(crate) type SparseSecondaryMap<K, V> = slotmap::SparseSecondaryMap<K, V, RandomState>;

thread_local! {
    pub(crate) static RT: Runtime = Default::default();
}

new_key_type! {
    pub(crate) struct ScopeId;
    pub(crate) struct SignalId;
    pub(crate) struct EffectId;
}

#[derive(Default)]
pub(crate) struct Runtime {
    pub observer: Cell<Option<EffectId>>,
    pub scopes: RefCell<SlotMap<ScopeId, RawScope>>,
    pub scope_parents: RefCell<SecondaryMap<ScopeId, ScopeId>>,
    pub scope_contexts: RefCell<SparseSecondaryMap<ScopeId, ScopeContexts>>,
    pub signals: RefCell<SlotMap<SignalId, RawSignal>>,
    pub signal_contexts: RefCell<SecondaryMap<SignalId, SignalContext>>,
    pub effects: RefCell<SlotMap<EffectId, RawEffect>>,
    pub effect_contexts: RefCell<SecondaryMap<EffectId, EffectContext>>,
}
