use crate::{scope::Scope, signal::Signal};
use std::marker::PhantomData;

pub trait StoreBuilder<'a> {
    type Store;
    fn build_store(cx: Scope<'a>, this: Self) -> Self::Store;
}

pub struct CreateDefault<T>(pub PhantomData<T>);

impl<T> Default for CreateDefault<T> {
    fn default() -> Self {
        CreateDefault(PhantomData)
    }
}

impl<'a, T: Default> StoreBuilder<'a> for CreateDefault<T> {
    type Store = T;

    fn build_store(_cx: Scope<'a>, _this: Self) -> Self::Store {
        T::default()
    }
}

#[derive(Default)]
pub struct CreateSelf<T>(pub T);

impl<'a, T> StoreBuilder<'a> for CreateSelf<T> {
    type Store = T;

    fn build_store(_cx: Scope<'a>, this: Self) -> Self::Store {
        this.0
    }
}

#[derive(Default)]
pub struct CreateSignal<T>(pub T);

impl<'a, T: 'static> StoreBuilder<'a> for CreateSignal<T> {
    type Store = Signal<'a, T>;

    fn build_store(cx: Scope<'a>, this: Self) -> Self::Store {
        cx.create_signal(this.0)
    }
}

impl<'a> Scope<'a> {
    pub fn create_store<T>(self, t: T) -> &'a T::Store
    where
        T: StoreBuilder<'a>,
    {
        self.create_variable(T::build_store(self, t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Builder {
        state: i32,
        data: String,
    }

    struct Store<'a> {
        state: Signal<'a, i32>,
        data: String,
    }

    impl<'a> StoreBuilder<'a> for Builder {
        type Store = Store<'a>;

        fn build_store(cx: Scope<'a>, this: Self) -> Self::Store {
            let Builder { state, data } = this;
            Store {
                state: cx.create_signal(state),
                data,
            }
        }
    }

    #[test]
    fn store_builder() {
        Scope::create_root(|cx| {
            let buidler = Builder {
                state: -1,
                data: String::from("xFrame"),
            };
            let store = cx.create_store(buidler);
            assert_eq!(*store.state.get(), -1);
            assert_eq!(&store.data, "xFrame");
        });
    }

    #[test]
    fn use_store_as_context() {
        Scope::create_root(|cx| {
            cx.provide_context(Builder {
                state: -1,
                ..Default::default()
            });
            cx.create_child(|cx| {
                let store = cx.use_context::<Builder>();
                assert_eq!(*store.state.get(), -1);
            });
        });
    }
}
