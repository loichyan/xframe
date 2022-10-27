use crate::{
    scope::{OwnedScope, ScopeInherited},
    shared::Empty,
    store::StoreBuilder,
};
use ahash::AHashMap;
use std::{any::TypeId, cell::RefCell, fmt, marker::PhantomData};

struct ContextId<T>(PhantomData<T>);

fn context_id<T: 'static>() -> TypeId {
    TypeId::of::<ContextId<T>>()
}

#[derive(Default)]
pub(crate) struct Contexts<'a> {
    inner: RefCell<ContextsInner<'a>>,
}

type ContextsInner<'a> = AHashMap<TypeId, &'a (dyn 'a + Empty)>;

impl fmt::Debug for Contexts<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.inner.borrow().keys()).finish()
    }
}

fn use_context_from<'a, T>(contexts: &ContextsInner<'a>) -> Option<&'a T::Store>
where
    T: 'static + StoreBuilder<'a>,
{
    contexts
        .get(&context_id::<T>())
        .copied()
        // SAFETY: The type is associated with `<T as Store>`, and this context
        // can only accessed from current and child scopes.
        .map(|any| unsafe { &*(any as *const dyn Empty as *const T::Store) })
}

fn use_context_from_ancestors<'a, T>(inherited: &'a ScopeInherited) -> Option<&'a T::Store>
where
    T: 'static + StoreBuilder<'a>,
{
    use_context_from::<T>(&inherited.contexts.inner.borrow())
        .or_else(|| inherited.parent.and_then(use_context_from_ancestors::<T>))
}

impl<'a> OwnedScope<'a> {
    pub fn try_provide_context<T>(&'a self, t: T) -> Result<&'a T::Store, &'a T::Store>
    where
        T: 'static + StoreBuilder<'a>,
    {
        let contexts = &mut self.inherited().contexts.inner.borrow_mut();
        if let Some(context) = use_context_from::<T>(contexts) {
            Err(context)
        } else {
            let context = self.create_store(t);
            contexts.insert(context_id::<T>(), context as &dyn Empty);
            Ok(context)
        }
    }

    pub fn provide_context<T>(&'a self, t: T) -> &'a T::Store
    where
        T: 'static + StoreBuilder<'a>,
    {
        self.try_provide_context(t)
            .unwrap_or_else(|_| panic!("context provided in current scope"))
    }

    pub fn try_use_context<T>(&'a self) -> Option<&'a T::Store>
    where
        T: 'static + StoreBuilder<'a>,
    {
        use_context_from_ancestors::<T>(self.inherited())
    }

    pub fn use_context<T>(&'a self) -> &'a T::Store
    where
        T: 'static + StoreBuilder<'a>,
    {
        self.try_use_context::<T>().expect("context not provided")
    }
}

#[cfg(test)]
mod tests {
    use crate::create_root;
    use crate::store::*;

    #[test]
    fn provide_and_use_context() {
        create_root(|cx| {
            cx.provide_context(CreateSignal(777i32));
            let x = cx.use_context::<CreateSignal<i32>>();
            assert_eq!(*x.get(), 777);
        });
    }

    #[test]
    fn use_context_from_child_scope() {
        create_root(|cx| {
            cx.provide_context(CreateSelf(777i32));
            cx.create_child(|cx| {
                let x = cx.use_context::<CreateSelf<i32>>();
                assert_eq!(*x, 777);
            });
        });
    }

    #[test]
    fn unique_context_in_same_scope() {
        create_root(|cx| {
            cx.provide_context(CreateSelf(777i32));
            assert!(cx.try_provide_context(CreateSelf(777i32)).is_err());
        });
    }
}
