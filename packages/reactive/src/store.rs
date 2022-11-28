use crate::{scope::Scope, signal::Signal, variable::Variable};
use std::marker::PhantomData;

/// A trait for constructing composite states.
///
/// This is useful when providing a non-`'static` lifetime bound type as a context.
/// The builder should be a `'static` type to idenfify a context.
pub trait StoreBuilder: 'static {
    type Store<'a>;
    fn build_store(self, cx: Scope<'_>) -> Self::Store<'_>;
}

pub struct CreateDefault<T: 'static>(pub PhantomData<T>);

impl<T> Default for CreateDefault<T> {
    fn default() -> Self {
        CreateDefault(PhantomData)
    }
}

impl<T: Default> StoreBuilder for CreateDefault<T> {
    type Store<'a> = T;

    fn build_store(self, _cx: Scope<'_>) -> Self::Store<'_> {
        T::default()
    }
}

#[derive(Default)]
pub struct CreateSelf<T: 'static>(pub T);

impl<T> StoreBuilder for CreateSelf<T> {
    type Store<'a> = T;

    fn build_store(self, _cx: Scope<'_>) -> Self::Store<'_> {
        self.0
    }
}

#[derive(Default)]
pub struct CreateSignal<T: 'static>(pub T);

impl<T> StoreBuilder for CreateSignal<T> {
    type Store<'a> = Signal<'a, T>;

    fn build_store(self, cx: Scope<'_>) -> Self::Store<'_> {
        cx.create_signal(self.0)
    }
}

impl<'a> Scope<'a> {
    pub fn create_store<T>(&self, t: T) -> Variable<'a, T::Store<'a>>
    where
        T: StoreBuilder,
    {
        self.create_variable(t.build_store(*self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_root;

    #[test]
    fn create_default() {
        struct Hello(String);
        impl Default for Hello {
            fn default() -> Self {
                Self(String::from("Hello, world!"))
            }
        }

        create_root(|cx| {
            let store = cx.create_store(CreateDefault::<Hello>::default());
            assert_eq!(&store.get().0, "Hello, world!");
        });
    }

    #[test]
    fn create_self() {
        create_root(|cx| {
            let store = cx.create_store(CreateSelf(-1));
            assert_eq!(*store.get(), -1);
        });
    }

    #[test]
    fn create_signal() {
        create_root(|cx| {
            let store = cx.create_store(CreateSignal(-1));
            let counter = cx.create_signal(0);
            cx.create_effect(move |_| {
                store.get().track();
                counter.update(|x| *x + 1);
            });
            assert_eq!(*counter.get(), 1);
            store.get().trigger();
            assert_eq!(*counter.get(), 2);
        });
    }

    #[derive(Default)]
    struct Builder {
        state: i32,
        data: String,
    }

    struct Store<'a> {
        state: Signal<'a, i32>,
        data: String,
    }

    impl StoreBuilder for Builder {
        type Store<'a> = Store<'a>;

        fn build_store(self, cx: Scope<'_>) -> Self::Store<'_> {
            let Builder { state, data } = self;
            Store {
                state: cx.create_signal(state),
                data,
            }
        }
    }

    #[test]
    fn store_builder() {
        create_root(|cx| {
            let buidler = Builder {
                state: -1,
                data: String::from("xFrame"),
            };
            let store = cx.create_store(buidler);
            assert_eq!(*store.get().state.get(), -1);
            assert_eq!(&store.get().data, "xFrame");
        });
    }

    #[test]
    fn use_store_as_context() {
        create_root(|cx| {
            cx.provide_context(Builder {
                state: -1,
                ..Default::default()
            });
            cx.create_child(|cx| {
                let store = cx.use_context::<Builder>();
                assert_eq!(*store.get().state.get(), -1);
            });
        });
    }

    #[test]
    fn signal_in_store() {
        create_root(|cx| {
            let builder = Builder {
                state: -1,
                data: String::from("xFrame"),
            };
            let store = cx.create_store(builder);
            let double = cx.create_memo(move || *store.get().state.get() * 2);
            assert_eq!(*double.get(), -2);
            store.get().state.set(1);
            assert_eq!(*double.get(), 2);
        });
    }
}
