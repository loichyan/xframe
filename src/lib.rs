#![warn(clippy::undocumented_unsafe_blocks)]

mod context;
mod effect;
mod memo;
mod scope;
mod shared;
mod signal;
mod store;
mod variable;

type InvariantLifetime<'a> = *mut &'a ();
type CovariantLifetime<'a> = *const &'a ();

trait Empty {}
impl<T> Empty for T {}

pub use effect::Effect;
pub use scope::{create_root, BoundedScope, Scope, ScopeDisposer};
pub use signal::{ReadSignal, Signal, SignalModify};
pub use store::{CreateDefault, CreateSelf, CreateSignal, StoreBuilder};
pub use variable::{VarRef, VarRefMut, Variable};

#[test]
fn readme_example() {
    use crate::*;

    create_root(|cx| {
        let state = cx.create_signal(1);

        let double = cx.create_memo(move || *state.get() * 2);
        assert_eq!(*double.get(), 2);

        state.set(2);
        assert_eq!(*double.get(), 4);

        state.set(3);
        assert_eq!(*double.get(), 6);
    });
}
