use crate::{
    scope::Scope,
    shared::{ScopeId, Shared, VariableId},
    store::StoreBuilder,
    variable::Variable,
    InvariantLifetime,
};
use ahash::AHashMap;
use std::{any::TypeId, marker::PhantomData};

#[derive(Default)]
pub(crate) struct ScopeContexts<'a> {
    content: AHashMap<TypeId, VariableId>,
    marker: PhantomData<InvariantLifetime<'a>>,
}

struct ContextId<T>(PhantomData<T>);

fn context_id<T: 'static>() -> TypeId {
    TypeId::of::<ContextId<T>>()
}

impl ScopeId {
    fn find_context<'a, T>(&self, shared: &Shared<'a>) -> Option<Variable<'a, T::Store<'a>>>
    where
        T: StoreBuilder,
    {
        shared
            .scope_contexts
            .borrow()
            .get(*self)
            .and_then(|contexts| {
                contexts
                    .content
                    .get(&context_id::<T>())
                    // SAFETY: This conversion is safe bacause:
                    // 1. The type is associated with `<T as StoreBuilder>` and
                    // stored as the key in a hashmap;
                    // 2. This context can only be accessed from current and child
                    // scopes as a `Variable` which is safe to use.
                    .map(|id| unsafe { id.bound() })
            })
    }

    fn find_context_recursive<'a, T>(
        &self,
        shared: &Shared<'a>,
    ) -> Option<Variable<'a, T::Store<'a>>>
    where
        T: StoreBuilder,
    {
        self.find_context::<T>(shared).or_else(|| {
            shared
                .scope_parents
                .borrow()
                .get(*self)
                .and_then(|parent| parent.find_context_recursive::<T>(shared))
        })
    }
}

impl<'a> Scope<'a> {
    /// Provide a context in current scope which is identified by the [`TypeId`]
    /// of given [`StoreBuilder`].
    ///
    /// # Panics
    ///
    /// Panics if the same context has already been provided.
    pub fn provide_context<T>(&self, t: T) -> Variable<'a, T::Store<'a>>
    where
        T: StoreBuilder,
    {
        self.try_provide_context(t)
            .unwrap_or_else(|_| panic!("tried to provide a duplicated context in the same scope"))
    }

    /// Provide a context in current scope which is identified by the [`TypeId`]
    /// of given [`StoreBuilder`]. If the same context has not been provided
    /// then return its reference, otherwise return the existing one as an error.
    pub fn try_provide_context<T>(
        &self,
        t: T,
    ) -> Result<Variable<'a, T::Store<'a>>, Variable<'a, T::Store<'a>>>
    where
        T: StoreBuilder,
    {
        self.with_shared(|shared| {
            if let Some(context) = self.id.find_context::<T>(shared) {
                Err(context)
            } else {
                let variable = self.create_store(t);
                shared
                    .scope_contexts
                    .borrow_mut()
                    .entry(self.id)
                    .unwrap_or_else(|| unreachable!())
                    .or_default()
                    .content
                    .insert(context_id::<T>(), variable.id);
                Ok(variable)
            }
        })
    }

    /// Loop up the context in the current and parent scopes accroding to the
    /// given builder type.
    ///
    /// # Panics
    ///
    /// Panics if the context is not provided.
    pub fn use_context<T>(&self) -> Variable<'a, T::Store<'a>>
    where
        T: StoreBuilder,
    {
        self.try_use_context::<T>()
            .unwrap_or_else(|| panic!("tried to use a nonexistent context"))
    }

    /// Loop up the context in the current and parent scopes accroding to the
    /// given builder type.
    pub fn try_use_context<T>(&self) -> Option<Variable<'a, T::Store<'a>>>
    where
        T: StoreBuilder,
    {
        self.with_shared(|shared| self.id.find_context_recursive::<T>(shared))
    }
}

#[cfg(test)]
mod tests {
    use crate::create_root;
    use crate::store::*;

    #[test]
    fn provide_and_use_context() {
        create_root(|cx| {
            cx.provide_context(CreateSignal(777));
            let x = cx.use_context::<CreateSignal<i32>>();
            assert_eq!(*x.get().get(), 777);
        });
    }

    #[test]
    fn use_context_from_child_scope() {
        create_root(|cx| {
            cx.provide_context(CreateSelf(777));
            cx.provide_context(CreateSelf("Hello, xFrame!"));
            cx.create_child(|cx| {
                let x = cx.use_context::<CreateSelf<i32>>();
                assert_eq!(*x.get(), 777);
                let s = cx.use_context::<CreateSelf<&str>>();
                assert_eq!(*s.get(), "Hello, xFrame!");
            });
        });
    }

    #[test]
    fn unique_context_in_same_scope() {
        create_root(|cx| {
            cx.provide_context(CreateSelf(777));
            assert_eq!(
                *cx.try_provide_context(CreateSelf(233)).unwrap_err().get(),
                777
            );
        });
    }

    #[test]
    #[should_panic = "tried to provide a duplicated context in the same scope"]
    fn panic_unique_context_in_same_scope() {
        create_root(|cx| {
            cx.provide_context(CreateSelf(777));
            cx.provide_context(CreateSelf(233));
        });
    }

    #[test]
    fn use_nonexistent_context() {
        create_root(|cx| {
            assert!(cx.try_use_context::<CreateSelf<i32>>().is_none());
        });
    }

    #[test]
    #[should_panic = "tried to use a nonexistent context"]
    fn panic_use_nonexistent_context() {
        create_root(|cx| {
            cx.use_context::<CreateSignal<i32>>();
        });
    }
}
