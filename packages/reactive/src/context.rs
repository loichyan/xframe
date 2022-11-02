use crate::{
    scope::Scope,
    shared::{ScopeId, Shared, VariableId},
    store::StoreBuilder,
    variable::Variable,
};
use ahash::AHashMap;
use std::{any::TypeId, marker::PhantomData};

#[derive(Default)]
pub(crate) struct ScopeContexts {
    content: AHashMap<TypeId, VariableId>,
}

struct ContextId<T>(PhantomData<T>);

fn context_id<T: 'static>() -> TypeId {
    TypeId::of::<ContextId<T>>()
}

impl ScopeId {
    fn find_context<'a, T>(&self, shared: &'a Shared) -> Option<Variable<'a, T::Store<'a>>>
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
                    .map(|id| unsafe { id.create_variable(shared) })
            })
    }

    fn find_context_recursive<'a, T>(
        &self,
        shared: &'a Shared,
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
    pub fn provide_context<T>(self, t: T) -> Variable<'a, T::Store<'a>>
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
        self,
        t: T,
    ) -> Result<Variable<'a, T::Store<'a>>, Variable<'a, T::Store<'a>>>
    where
        T: StoreBuilder,
    {
        if let Some(context) = self.id.find_context::<T>(self.shared) {
            Err(context)
        } else {
            let variable = self.create_store(t);
            self.shared
                .scope_contexts
                .borrow_mut()
                .entry(self.id)
                .unwrap_or_else(|| unreachable!())
                .or_default()
                .content
                .insert(context_id::<T>(), variable.id);
            Ok(variable)
        }
    }

    /// Loop up the context in the current and parent scopes accroding to the
    /// given builder type.
    ///
    /// # Panics
    ///
    /// Panics if the context is not provided.
    pub fn use_context<T>(self) -> Variable<'a, T::Store<'a>>
    where
        T: StoreBuilder,
    {
        self.try_use_context::<T>()
            .unwrap_or_else(|| panic!("tried to use a nonexistent context"))
    }

    /// Loop up the context in the current and parent scopes accroding to the
    /// given builder type.
    pub fn try_use_context<T>(self) -> Option<Variable<'a, T::Store<'a>>>
    where
        T: StoreBuilder,
    {
        self.id.find_context_recursive::<T>(self.shared)
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
            assert_eq!(*x.get().get(), 777);
        });
    }

    #[test]
    fn use_context_from_child_scope() {
        create_root(|cx| {
            cx.provide_context(CreateSelf(777i32));
            cx.create_child(|cx| {
                let x = cx.use_context::<CreateSelf<i32>>();
                assert_eq!(*x.get(), 777);
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
