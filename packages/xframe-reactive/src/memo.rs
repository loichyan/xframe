use crate::{
    scope::Scope,
    signal::{Modify, Signal},
};
use std::cell::Cell;

impl<'a> Scope<'a> {
    fn create_memo_impl<T: 'static>(
        self,
        mut f: impl 'a + FnMut() -> T,
        mut update: impl 'a + FnMut(T, Signal<'a, T>),
    ) -> Signal<'a, T> {
        let memo = self.create_variable(Cell::new(None::<Signal<'a, T>>));
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

    pub fn create_memo<T: 'static>(self, f: impl 'a + FnMut() -> T) -> Signal<'a, T> {
        self.create_memo_impl(f, |new_val, memo| memo.set(new_val))
    }

    pub fn create_seletor<T: 'static>(self, f: impl 'a + FnMut() -> T) -> Signal<'a, T>
    where
        T: PartialEq,
    {
        self.create_seletor_with(f, T::eq)
    }

    pub fn create_seletor_with<T: 'static>(
        self,
        f: impl 'a + FnMut() -> T,
        mut is_equal: impl 'a + FnMut(&T, &T) -> bool,
    ) -> Signal<'a, T> {
        self.create_memo_impl(f, move |new_val, memo| {
            let mut modify_memo = memo.modify();
            if !is_equal(&*modify_memo, &new_val) {
                *modify_memo = new_val;
                return;
            }
            Modify::drop_silent(modify_memo);
        })
    }
}
