mod context;
mod effect;
mod memo;
mod scope;
mod shared;
mod signal;
mod store;
mod variable;

type InvariantLifetime<'a> = &'a mut &'a mut ();
type CovariantLifetime<'a> = &'a mut ();

trait Empty {}
impl<T> Empty for T {}

pub use effect::Effect;
pub use scope::{create_root, BoundedScope, Scope, ScopeDisposer};
pub use signal::{Signal, SignalModify};
pub use store::{CreateDefault, CreateSelf, CreateSignal, StoreBuilder};
pub use variable::{VarRef, VarRefMut, Variable};
