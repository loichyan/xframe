mod context;
mod effect;
mod memo;
mod runtime;
mod scope;
mod selector;
mod signal;

type ThreadLocal = *const ();

trait Empty {}
impl<T> Empty for T {}

#[doc(inline)]
pub use {
    effect::Effect,
    scope::{create_root, untrack, Scope, ScopeDisposer},
    signal::{ReadSignal, Signal},
};

#[test]
fn readme_example() {
    use crate::*;

    create_root(|cx| {
        let state = cx.create_signal(1);

        let double = cx.create_memo(move || state.get() * 2);
        assert_eq!(double.get(), 2);

        state.set(2);
        assert_eq!(double.get(), 4);

        state.set(3);
        assert_eq!(double.get(), 6);
    });
}
