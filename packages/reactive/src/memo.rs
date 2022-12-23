use crate::{
    scope::Scope,
    signal::{ReadSignal, Signal},
};
use std::{cell::Cell, rc::Rc};

impl Scope {
    fn create_memo_impl<T: 'static>(
        &self,
        mut f: impl 'static + FnMut() -> T,
        mut update: impl 'static + FnMut(T, Signal<T>),
    ) -> ReadSignal<T> {
        let memo = Rc::new(Cell::new(None::<Signal<T>>));
        let cx = *self;
        self.create_effect({
            let memo = memo.clone();
            move || {
                let new_val = f();
                if let Some(signal) = memo.get() {
                    update(new_val, signal);
                } else {
                    let signal = cx.create_signal(new_val);
                    memo.set(Some(signal));
                }
            }
        });
        memo.get().unwrap().into()
    }

    pub fn create_memo<T: 'static>(&self, f: impl 'static + FnMut() -> T) -> ReadSignal<T> {
        self.create_memo_impl(f, |new_val, memo| memo.set(new_val))
    }

    pub fn create_selector<T>(&self, f: impl 'static + FnMut() -> T) -> ReadSignal<T>
    where
        T: 'static + PartialEq,
    {
        self.create_selector_with(f, T::eq)
    }

    pub fn create_selector_with<T: 'static>(
        &self,
        f: impl 'static + FnMut() -> T,
        mut is_equal: impl 'static + FnMut(&T, &T) -> bool,
    ) -> ReadSignal<T> {
        self.create_memo_impl(f, move |new_val, memo| {
            let mut updated = false;
            memo.write_slient(|old_val| {
                if !is_equal(old_val, &new_val) {
                    *old_val = new_val;
                    updated = true;
                }
            });
            if updated {
                memo.trigger();
            }
        })
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

    #[test]
    fn selector() {
        create_root(|cx| {
            let state = cx.create_signal(1);
            let double = cx.create_selector(move || state.get() * 2);

            let counter2 = cx.create_signal(0);
            cx.create_effect(move || {
                double.track();
                counter2.update(|x| x + 1);
            });
            assert_eq!(counter2.get(), 1);
            assert_eq!(double.get(), 2);

            state.set(2);
            assert_eq!(counter2.get(), 2);
            assert_eq!(double.get(), 4);

            // Equal updates should be ignored.
            state.set(2);
            assert_eq!(counter2.get(), 2);
            assert_eq!(double.get(), 4);
        });
    }
}
