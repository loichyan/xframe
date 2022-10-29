use crate::{
    scope::{OwnedScope, Scope},
    OwnedReadSignal, OwnedSignal, Variable,
};
use std::marker::PhantomData;

/// A trait for constructing composite states.
///
/// This is useful when providing a non-`'static` lifetime bound type as a context.
/// The builder should be a `'static` type to idenfify a context.
pub trait StoreBuilder<'a>: 'static {
    type Store;
    fn build_store(self, cx: Scope<'a>) -> Self::Store;
}

pub struct CreateDefault<T: 'static>(pub PhantomData<T>);

impl<T> Default for CreateDefault<T> {
    fn default() -> Self {
        CreateDefault(PhantomData)
    }
}

impl<'a, T: Default> StoreBuilder<'a> for CreateDefault<T> {
    type Store = T;

    fn build_store(self, _cx: Scope<'a>) -> Self::Store {
        T::default()
    }
}

#[derive(Default)]
pub struct CreateSelf<T: 'static>(pub T);

impl<'a, T> StoreBuilder<'a> for CreateSelf<T> {
    type Store = T;

    fn build_store(self, _cx: Scope<'a>) -> Self::Store {
        self.0
    }
}

#[derive(Default)]
pub struct CreateSignal<T: 'static>(pub T);

impl<'a, T> StoreBuilder<'a> for CreateSignal<T> {
    type Store = OwnedSignal<'a, T>;

    fn build_store(self, cx: Scope<'a>) -> Self::Store {
        cx.create_owned_signal(self.0)
    }
}

#[derive(Default)]
pub struct CreateReadSignal<T: 'static>(pub T);

impl<'a, T> StoreBuilder<'a> for CreateReadSignal<T> {
    type Store = OwnedReadSignal<'a, T>;

    fn build_store(self, cx: Scope<'a>) -> Self::Store {
        cx.create_owned_read_signal(self.0)
    }
}

impl<'a> OwnedScope<'a> {
    pub fn create_store<T>(&'a self, t: T) -> Variable<'a, T::Store>
    where
        T: StoreBuilder<'a>,
    {
        self.create_variable(t.build_store(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_root;

    #[derive(Default)]
    struct Builder {
        state: i32,
        data: String,
    }

    struct Store<'a> {
        state: OwnedSignal<'a, i32>,
        data: String,
    }

    impl<'a> StoreBuilder<'a> for Builder {
        type Store = Store<'a>;

        fn build_store(self, cx: Scope<'a>) -> Self::Store {
            let Builder { state, data } = self;
            Store {
                state: cx.create_owned_signal(state),
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
    fn reactive_store() {
        create_root(|cx| {
            let builder = Builder {
                state: -1,
                data: String::from("xFrame"),
            };
            let store = cx.create_store(builder);
            let double = cx.create_memo(|| *store.get().state.get() * 2);
            assert_eq!(*double.get(), -2);
            store.get().state.set(1);
            assert_eq!(*double.get(), 2);
        });
    }
}
