use crate::{
    scope::OwnedScope,
    signal::{Signal, SignalModify},
};
use std::cell::Cell;

impl<'a> OwnedScope<'a> {
    fn create_memo_impl<T>(
        &'a self,
        mut f: impl 'a + FnMut() -> T,
        mut update: impl 'a + FnMut(T, Signal<'a, T>),
    ) -> Signal<'a, T> {
        // SAFETY: An `Option` will never read it's underlying value when getting dropped.
        let memo = unsafe { self.create_variable_unchecked(Cell::new(None::<Signal<'a, T>>)) };
        self.create_effect(move |_| {
            let new_val = f();
            if let Some(signal) = memo.get() {
                update(new_val, signal);
            } else {
                let signal = self.create_signal(new_val);
                memo.set(Some(signal));
            }
        });
        memo.get().unwrap()
    }

    pub fn create_memo<T>(&'a self, f: impl 'a + FnMut() -> T) -> Signal<'a, T> {
        self.create_memo_impl(f, |new_val, memo| memo.set(new_val))
    }

    pub fn create_seletor<T>(&'a self, f: impl 'a + FnMut() -> T) -> Signal<'a, T>
    where
        T: PartialEq,
    {
        self.create_seletor_with(f, T::eq)
    }

    pub fn create_seletor_with<T>(
        &'a self,
        f: impl 'a + FnMut() -> T,
        mut is_equal: impl 'a + FnMut(&T, &T) -> bool,
    ) -> Signal<'a, T> {
        self.create_memo_impl(f, move |new_val, memo| {
            let mut modify_memo = memo.modify();
            if !is_equal(&*modify_memo, &new_val) {
                *modify_memo = new_val;
                return;
            }
            SignalModify::drop_silent(modify_memo);
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::create_root;

    #[test]
    fn reactive_memo() {
        create_root(|cx| {
            let state = cx.create_signal(1);

            let double = cx.create_memo(|| *state.get() * 2);
            assert_eq!(*double.get(), 2);

            state.set(2);
            assert_eq!(*double.get(), 4);

            state.set(3);
            assert_eq!(*double.get(), 6);
        });
    }

    #[test]
    fn memo_only_run_when_triggered() {
        create_root(|cx| {
            let state = cx.create_signal(1);
            let counter = cx.create_signal(0);

            let double = cx.create_memo(|| {
                counter.update(|x| *x + 1);
                *state.get() * 2
            });
            assert_eq!(*double.get(), 2);
            assert_eq!(*counter.get(), 1);

            state.set(2);
            assert_eq!(*double.get(), 4);
            assert_eq!(*counter.get(), 2);

            assert_eq!(*double.get(), 4);
            assert_eq!(*counter.get(), 2);
        });
    }

    #[test]
    fn memo_on_memo() {
        create_root(|cx| {
            let state = cx.create_signal(1);
            let double = cx.create_memo(|| *state.get() * 2);
            let quad = cx.create_memo(|| *double.get() * 2);

            assert_eq!(*quad.get(), 4);
            state.set(2);
            assert_eq!(*quad.get(), 8);
        });
    }

    #[test]
    fn untracked_memo() {
        create_root(|cx| {
            let state = cx.create_signal(1);
            let double = cx.create_memo(|| *state.get_untracked() * 2);

            assert_eq!(*double.get(), 2);
            state.set(2);
            assert_eq!(*double.get(), 2);
        });
    }

    #[test]
    fn reactive_selector() {
        create_root(|cx| {
            let state = cx.create_signal(1);
            let double = cx.create_seletor(|| *state.get() * 2);

            let counter2 = cx.create_signal(0);
            cx.create_effect(|_| {
                double.track();
                counter2.update(|x| *x + 1);
            });
            assert_eq!(*counter2.get(), 1);
            assert_eq!(*double.get(), 2);

            state.set(2);
            assert_eq!(*counter2.get(), 2);
            assert_eq!(*double.get(), 4);

            state.set(2);
            assert_eq!(*counter2.get(), 2);
            assert_eq!(*double.get(), 4);
        });
    }
}
