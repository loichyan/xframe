use crate::{
    scope::Scope,
    signal::{ReadSignal, Signal},
};
use std::{cell::Cell, rc::Rc};

impl Scope {
    pub fn create_memo<T: 'static + PartialEq>(
        &self,
        f: impl 'static + FnMut() -> T,
    ) -> ReadSignal<T> {
        self.create_memo_with(f, T::eq)
    }

    pub fn create_memo_with<T: 'static>(
        &self,
        mut f: impl 'static + FnMut() -> T,
        mut is_equal: impl 'static + FnMut(&T, &T) -> bool,
    ) -> ReadSignal<T> {
        let memo = Rc::new(Cell::new(None::<Signal<T>>));
        let cx = *self;
        self.create_effect({
            let memo = memo.clone();
            move || {
                let new_val = f();
                if let Some(signal) = memo.get() {
                    let mut updated = false;
                    signal.write_slient(|old| {
                        if !is_equal(old, &new_val) {
                            *old = new_val;
                            updated = true;
                        }
                    });
                    if updated {
                        signal.trigger();
                    }
                } else {
                    let signal = cx.create_signal(new_val);
                    memo.set(Some(signal));
                }
            }
        });
        memo.get().unwrap().into()
    }
}

#[cfg(test)]
mod tests {
    use crate::create_root;

    #[test]
    fn memo() {
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

    #[test]
    fn memo_only_run_when_triggered() {
        create_root(|cx| {
            let state = cx.create_signal(1);
            let counter = cx.create_signal(0);

            let double = cx.create_memo(move || {
                counter.update(|x| x + 1);
                state.get() * 2
            });
            assert_eq!(double.get(), 2);
            assert_eq!(counter.get(), 1);

            state.set(2);
            assert_eq!(double.get(), 4);
            assert_eq!(counter.get(), 2);

            assert_eq!(double.get(), 4);
            assert_eq!(counter.get(), 2);
        });
    }

    #[test]
    fn memo_on_memo() {
        create_root(|cx| {
            let state = cx.create_signal(1);
            let double = cx.create_memo(move || state.get() * 2);
            let quad = cx.create_memo(move || double.get() * 2);

            assert_eq!(quad.get(), 4);
            state.set(2);
            assert_eq!(quad.get(), 8);
        });
    }

    #[test]
    fn untracked_memo() {
        create_root(|cx| {
            let state = cx.create_signal(1);
            let double = cx.create_memo(move || state.get_untracked() * 2);

            assert_eq!(double.get(), 2);
            state.set(2);
            assert_eq!(double.get(), 2);
        });
    }
}
