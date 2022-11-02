use crate::{scope::Scope, signal::Signal};

impl<'a> Scope<'a> {
    fn create_memo_impl<T: 'a>(
        self,
        mut f: impl 'a + FnMut() -> T,
        mut update: impl 'a + FnMut(T, Signal<'a, T>),
    ) -> Signal<'a, T> {
        let memo = self.create_cell(None::<Signal<'a, T>>);
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

    pub fn create_memo<T: 'a>(self, f: impl 'a + FnMut() -> T) -> Signal<'a, T> {
        self.create_memo_impl(f, |new_val, memo| memo.set(new_val))
    }

    pub fn create_seletor<T: 'a>(self, f: impl 'a + FnMut() -> T) -> Signal<'a, T>
    where
        T: PartialEq,
    {
        self.create_seletor_with(f, T::eq)
    }

    pub fn create_seletor_with<T: 'a>(
        self,
        f: impl 'a + FnMut() -> T,
        mut is_equal: impl 'a + FnMut(&T, &T) -> bool,
    ) -> Signal<'a, T> {
        self.create_memo_impl(f, move |new_val, memo| {
            let modify_memo = &mut *memo.get_mut();
            if !is_equal(modify_memo, &new_val) {
                *modify_memo = new_val;
                memo.trigger();
                return;
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

            let double = cx.create_memo(move || *state.get() * 2);
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

            let double = cx.create_memo(move || {
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
            let double = cx.create_memo(move || *state.get() * 2);
            let quad = cx.create_memo(move || *double.get() * 2);

            assert_eq!(*quad.get(), 4);
            state.set(2);
            assert_eq!(*quad.get(), 8);
        });
    }

    #[test]
    fn untracked_memo() {
        create_root(|cx| {
            let state = cx.create_signal(1);
            let double = cx.create_memo(move || *state.get_untracked() * 2);

            assert_eq!(*double.get(), 2);
            state.set(2);
            assert_eq!(*double.get(), 2);
        });
    }

    #[test]
    fn selector() {
        create_root(|cx| {
            let state = cx.create_signal(1);
            let double = cx.create_seletor(move || *state.get() * 2);

            let counter2 = cx.create_signal(0);
            cx.create_effect(move |_| {
                double.track();
                counter2.update(|x| *x + 1);
            });
            assert_eq!(*counter2.get(), 1);
            assert_eq!(*double.get(), 2);

            state.set(2);
            assert_eq!(*counter2.get(), 2);
            assert_eq!(*double.get(), 4);

            // Equal updates should be ignored.
            state.set(2);
            assert_eq!(*counter2.get(), 2);
            assert_eq!(*double.get(), 4);
        });
    }
}
