mod context;
mod effect;
mod memo;
mod scope;
mod shared;
mod signal;
mod variable;

type ThreadLocal = *const ();

trait Empty {}
impl<T> Empty for T {}

pub use effect::Effect;
pub use scope::{create_root, Scope, ScopeDisposer};
pub use signal::{ReadSignal, Signal};
pub use variable::Variable;

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
