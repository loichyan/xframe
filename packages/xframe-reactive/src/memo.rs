use crate::{
    scope::Scope,
    signal::{Modify, ReadSignal, Signal},
};
use std::cell::Cell;

fn create_memo_impl<'a, T>(
    cx: Scope<'a>,
    mut f: impl 'a + FnMut() -> T,
    mut update: impl 'a + FnMut(T, &'a Signal<T>),
) -> &'a ReadSignal<T> {
    let memo = cx.create_variable(Cell::new(None::<&'a Signal<T>>));
    cx.create_effect(move |_| {
        let new_val = f();
        if let Some(signal) = memo.get() {
            update(new_val, signal);
        } else {
            let signal = cx.create_signal(new_val);
            memo.set(Some(signal));
        }
    });
    memo.get().unwrap()
}

impl<'a> Scope<'a> {
    pub fn create_memo<T>(self, f: impl 'a + FnMut() -> T) -> &'a ReadSignal<T> {
        create_memo_impl(self, f, |new_val, memo| memo.set(new_val))
    }

    pub fn create_seletor<T, F>(self, f: impl 'a + FnMut() -> T) -> &'a ReadSignal<T>
    where
        T: PartialEq,
    {
        self.create_seletor_with(f, T::eq)
    }

    pub fn create_seletor_with<T>(
        self,
        f: impl 'a + FnMut() -> T,
        mut is_equal: impl 'a + FnMut(&T, &T) -> bool,
    ) -> &'a ReadSignal<T> {
        create_memo_impl(self, f, move |new_val, memo| {
            let mut modify_memo = memo.modify();
            if !is_equal(&*modify_memo, &new_val) {
                *modify_memo = new_val;
                return;
            }
            Modify::drop_silent(modify_memo);
        })
    }
}
