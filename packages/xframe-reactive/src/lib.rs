mod arena;
mod context;
mod effect;
mod memo;
mod scope;
mod signal;
mod store;

type InvariantLifetime<'a> = &'a mut &'a mut ();
type CovariantLifetime<'a> = &'a ();

trait Empty {}
impl<T> Empty for T {}

pub use effect::Effect;
pub use scope::{BoundedScope, Scope, ScopeDisposer};
pub use signal::{Modify, Ref, Signal};
pub use store::{CreateDefault, CreateSelf, CreateSignal, StoreBuilder};
