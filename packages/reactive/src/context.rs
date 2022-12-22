use crate::{
    runtime::{Runtime, ScopeId, SignalId, RT},
    scope::Scope,
    signal::{ReadSignal, Signal},
};
use ahash::AHashMap;
use std::{any::TypeId, marker::PhantomData};

#[derive(Default)]
pub(crate) struct ScopeContexts {
    content: AHashMap<TypeId, SignalId>,
}

struct ContextId<T>(PhantomData<T>);

fn context_id<T: 'static>() -> TypeId {
    TypeId::of::<ContextId<T>>()
}

impl ScopeId {
    fn find_context<T>(&self, rt: &Runtime) -> Option<Signal<T>> {
        rt.scope_contexts.borrow().get(*self).and_then(|contexts| {
            contexts.content.get(&context_id::<T>()).map(|&id| {
                Signal(ReadSignal {
                    id,
                    marker: PhantomData,
                })
            })
        })
    }

    fn find_context_recursive<T>(&self, rt: &Runtime) -> Option<Signal<T>> {
        self.find_context::<T>(rt).or_else(|| {
            rt.scope_parents
                .borrow()
                .get(*self)
                .and_then(|parent| parent.find_context_recursive::<T>(rt))
        })
    }
}

impl Scope {
    /// Provide a context in current scope which is identified by the [`TypeId`].
    ///
    /// # Panics
    ///
    /// Panics if the same context has already been provided.
    pub fn provide_context<T>(&self, t: T) -> Signal<T> {
        self.try_provide_context(t)
            .unwrap_or_else(|_| panic!("tried to provide a duplicated context in the same scope"))
    }

    /// Provide a context in current scope which is identified by the [`TypeId`].
    /// If the same context has not been provided then return its reference,
    /// otherwise return the existing one as an error.
    pub fn try_provide_context<T>(&self, t: T) -> Result<Signal<T>, Signal<T>> {
        RT.with(|rt| {
            if let Some(val) = self.id.find_context::<T>(rt) {
                Err(val)
            } else {
                let val = self.create_signal(t);
                rt.scope_contexts
                    .borrow_mut()
                    .entry(self.id)
                    .unwrap()
                    .or_default()
                    .content
                    .insert(context_id::<T>(), val.0.id);
                Ok(val)
            }
        })
    }

    /// Loop up the context in the current and parent scopes.
    ///
    /// # Panics
    ///
    /// Panics if the context is not provided.
    pub fn use_context<T>(&self) -> Signal<T> {
        self.try_use_context::<T>()
            .unwrap_or_else(|| panic!("tried to use a nonexistent context"))
    }

    /// Loop up the context in the current and parent scopes.
    pub fn try_use_context<T>(&self) -> Option<Signal<T>> {
        RT.with(|rt| self.id.find_context_recursive::<T>(rt))
    }
}

#[cfg(test)]
mod tests {
    use crate::create_root;

    #[test]
    fn provide_and_use_context() {
        create_root(|cx| {
            cx.provide_context(777);
            let x = cx.use_context::<i32>();
            assert_eq!(x.get(), 777);
        });
    }

    #[test]
    fn use_context_from_child_scope() {
        create_root(|cx| {
            cx.provide_context(777);
            cx.provide_context("Hello, xFrame!");
            cx.create_child(|cx| {
                let x = cx.use_context::<i32>();
                assert_eq!(x.get(), 777);
                let s = cx.use_context::<&str>();
                assert_eq!(s.get(), "Hello, xFrame!");
            });
        });
    }

    #[test]
    fn unique_context_in_same_scope() {
        create_root(|cx| {
            cx.provide_context(777);
            assert_eq!(cx.try_provide_context(233).unwrap_err().get(), 777);
        });
    }

    #[test]
    #[should_panic = "tried to provide a duplicated context in the same scope"]
    fn panic_unique_context_in_same_scope() {
        create_root(|cx| {
            cx.provide_context(777);
            cx.provide_context(233);
        });
    }

    #[test]
    #[should_panic = "tried to use a nonexistent context"]
    fn use_nonexistent_context() {
        create_root(|cx| {
            cx.use_context::<i32>();
        });
    }
}
