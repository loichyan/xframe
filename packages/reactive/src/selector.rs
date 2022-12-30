use crate::{
    scope::Scope,
    signal::{ReadSignal, Signal},
};
use std::{cell::RefCell, rc::Rc};

fn update_selector(signal: &Signal<bool>, selected: bool) {
    let mut updated = false;
    signal.write_slient(|t| {
        if *t != selected {
            *t = selected;
            updated = true;
        }
    });
    if updated {
        signal.trigger();
    }
}

impl Scope {
    pub fn create_selector<T>(
        &self,
        f: impl 'static + FnMut() -> T,
    ) -> impl Clone + Fn(T) -> ReadSignal<bool>
    where
        T: 'static + PartialEq,
    {
        self.create_selector_with(f, T::eq)
    }

    pub fn create_selector_with<T>(
        &self,
        mut f: impl 'static + FnMut() -> T,
        select: impl 'static + Clone + Fn(&T, &T) -> bool,
    ) -> impl Clone + Fn(T) -> ReadSignal<bool>
    where
        T: 'static + PartialEq,
    {
        let cx = *self;
        let selectors = Rc::new(RefCell::new(Vec::default()));
        let current = Rc::new(RefCell::new(None));

        cx.create_effect({
            let select = select.clone();
            let selectors = selectors.clone();
            let current = current.clone();
            move || {
                let new = f();
                if let Some(current) = current.borrow().as_ref() {
                    if current == &new {
                        return;
                    }
                    for (val, signal) in selectors.borrow().iter() {
                        if select(val, &new) {
                            update_selector(signal, true);
                        } else {
                            update_selector(signal, false);
                        }
                    }
                }
                *current.borrow_mut() = Some(new);
            }
        });

        move |new| {
            let signal = cx.create_signal(select(&new, current.borrow().as_ref().unwrap()));
            selectors.borrow_mut().push((new, signal));
            signal.into()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::create_root;

    #[test]
    fn selector() {
        create_root(|cx| {
            let state = cx.create_signal(0);
            let is_selected = cx.create_selector(move || state.get());

            let counter = cx.create_signal(0);
            let counter_false = cx.create_signal(0);
            cx.create_effect({
                let is_0 = is_selected(0);
                move || {
                    if is_0.get() {
                        counter.update(|x| x + 1);
                    } else {
                        counter_false.update(|x| x + 1);
                    }
                }
            });
            assert_eq!(counter.get(), 1);
            assert_eq!(counter_false.get(), 0);

            cx.create_effect({
                let is_5 = is_selected(5);
                move || {
                    if is_5.get() {
                        counter.update(|x| x + 1);
                    } else {
                        counter_false.update(|x| x + 1);
                    }
                }
            });
            assert_eq!(counter.get(), 1);
            assert_eq!(counter_false.get(), 1);

            state.set(5);
            assert_eq!(counter.get(), 2);
            assert_eq!(counter_false.get(), 2);

            state.set(0);
            assert_eq!(counter.get(), 3);
            assert_eq!(counter_false.get(), 3);
        });
    }

    #[test]
    fn selector_with() {
        create_root(|cx| {
            let state = cx.create_signal(1);
            let is_less_than = cx.create_selector_with(move || state.get(), |max, val| val < max);

            let counter_lt2 = cx.create_signal(0);
            cx.create_effect({
                let lt_2 = is_less_than(2);
                move || {
                    if lt_2.get() {
                        counter_lt2.update(|x| x + 1);
                    }
                }
            });
            assert_eq!(counter_lt2.get(), 1);

            let counter_lt5 = cx.create_signal(0);
            cx.create_effect({
                let lt_5 = is_less_than(5);
                move || {
                    if lt_5.get() {
                        counter_lt5.update(|x| x + 1);
                    }
                }
            });
            assert_eq!(counter_lt5.get(), 1);

            state.set(6);
            assert_eq!(counter_lt2.get(), 1);
            assert_eq!(counter_lt5.get(), 1);

            state.set(4);
            assert_eq!(counter_lt2.get(), 1);
            assert_eq!(counter_lt5.get(), 2);
        });
    }
}
