use crate::{scope::ScopeInherited, store::Store, Scope, Signal};
use ahash::AHashMap;
use std::{
    any::{Any, TypeId},
    cell::RefCell,
    fmt,
};

#[derive(Default)]
pub(crate) struct Contexts<'a> {
    inner: RefCell<AHashMap<TypeId, &'a dyn Any>>,
}

impl fmt::Debug for Contexts<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.inner.borrow().keys()).finish()
    }
}

fn type_id<T: 'static>() -> TypeId {
    TypeId::of::<Signal<T>>()
}

fn use_context_impl<'a, T: 'static>(inherited: &'a ScopeInherited) -> Option<&'a T> {
    if let Some(any) = inherited
        .contexts
        .inner
        .borrow()
        .get(&type_id::<T>())
        .copied()
    {
        Some(downcast_context(any))
    } else {
        inherited.parent.and_then(use_context_impl)
    }
}

fn downcast_context<T: 'static>(any: &dyn Any) -> &T {
    any.downcast_ref::<T>().unwrap_or_else(|| unreachable!())
}

impl<'a> Scope<'a> {
    pub fn try_provide_context<T>(self, input: T::Input) -> Result<&'a T, &'a T>
    where
        T: 'static + Store,
    {
        let store = self.create_store(input);
        if let Some(prev) = self
            .inherited()
            .contexts
            .inner
            .borrow_mut()
            .insert(type_id::<T>(), store as &dyn Any)
        {
            Err(downcast_context(prev))
        } else {
            Ok(store)
        }
    }

    pub fn provide_context<T>(self, input: T::Input) -> &'a T
    where
        T: 'static + Store,
    {
        self.try_provide_context(input)
            .unwrap_or_else(|_| panic!("context provided in current scope"))
    }

    pub fn try_use_context<T: 'static>(self) -> Option<&'a T> {
        use_context_impl(self.inherited())
    }

    pub fn use_context<T: 'static>(self) -> &'a T {
        self.try_use_context().expect("context not provided")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::PlainStore;

    #[test]
    fn provide_and_use_context() {
        Scope::create_root(|cx| {
            cx.provide_context::<PlainStore<_>>(777i32);
            let x = cx.use_context::<PlainStore<i32>>();
            assert_eq!(**x, 777);
        });
    }

    #[test]
    fn use_context_from_child_scope() {
        Scope::create_root(|cx| {
            cx.provide_context::<PlainStore<_>>(777i32);
            cx.create_child(|cx| {
                let x = cx.use_context::<PlainStore<i32>>();
                assert_eq!(**x, 777);
            });
        });
    }

    #[test]
    fn unique_context_in_same_scope() {
        Scope::create_root(|cx| {
            cx.provide_context::<PlainStore<_>>(777i32);
            assert!(cx.try_provide_context::<PlainStore<_>>(777i32).is_err());
        });
    }
}
