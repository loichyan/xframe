use crate::{
    scope::Scope,
    signal::{Modify, Signal},
};
use std::cell::Cell;

impl<'a> Scope<'a> {
    pub fn create_memo<T, F>(self, f: F) -> Signal<'a, T>
    where
        F: 'a + FnMut() -> T,
    {
        self.create_seletor_with(f, |_, _| false)
    }

    pub fn create_seletor<T, F>(self, f: F) -> Signal<'a, T>
    where
        T: PartialEq,
        F: 'a + FnMut() -> T,
    {
        self.create_seletor_with(f, T::eq)
    }

    pub fn create_seletor_with<T, F, C>(self, mut f: F, mut is_equal: C) -> Signal<'a, T>
    where
        F: 'a + FnMut() -> T,
        C: 'a + FnMut(&T, &T) -> bool,
    {
        let memo = &*self.create_variable(Cell::new(None::<Signal<T>>));
        self.create_effect(move |_| {
            let new_val = f();
            if let Some(signal) = memo.get() {
                let mut modify_memo = signal.modify();
                if !is_equal(&*modify_memo, &new_val) {
                    *modify_memo = new_val;
                    return;
                }
                Modify::drop_silent(modify_memo);
            } else {
                let signal = self.create_signal(new_val);
                memo.set(Some(signal));
            }
        });
        memo.get().unwrap()
    }
}
