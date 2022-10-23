use crate::{scope::ScopeInherited, store::Store, Scope};
use ahash::AHashMap;
use std::{any::TypeId, cell::RefCell, fmt};

type ContextsInner<'a> = AHashMap<TypeId, &'a (dyn 'a + Empty)>;

trait Empty {}
impl<T> Empty for T {}

#[derive(Default)]
pub(crate) struct Contexts<'a> {
    inner: RefCell<ContextsInner<'a>>,
}

impl fmt::Debug for Contexts<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.inner.borrow().keys()).finish()
    }
}

fn use_source_from<'a, T>(contexts: &ContextsInner<'a>) -> Option<&'a T::Source>
where
    T: 'static + Store<'a>,
{
    contexts
        .get(&TypeId::of::<T>())
        .copied()
        // SAFETY: The type is associated with `<T as Store>`, and this context
        // can only accessed from current and child scopes.
        .map(|any| unsafe { &*(any as *const dyn Empty as *const T::Source) })
}

fn use_context_impl<'a, T>(inherited: &'a ScopeInherited) -> Option<&'a T::Source>
where
    T: 'static + Store<'a>,
{
    use_source_from::<T>(&inherited.contexts.inner.borrow())
        .or_else(|| inherited.parent.and_then(use_context_impl::<T>))
}

impl<'a> Scope<'a> {
    pub fn try_provide_context<T>(self, t: T) -> Result<T::Output, T::Output>
    where
        T: 'static + Store<'a>,
    {
        let contexts = &mut self.inherited().contexts.inner.borrow_mut();
        if let Some(source) = use_source_from::<T>(contexts) {
            Err(T::make_output(self, source))
        } else {
            let source = self.create_variable(T::create_source(self, t));
            contexts.insert(TypeId::of::<T>(), source as &dyn Empty);
            Ok(T::make_output(self, source))
        }
    }

    pub fn provide_context<T>(self, t: T) -> T::Output
    where
        T: 'static + Store<'a>,
    {
        self.try_provide_context(t)
            .unwrap_or_else(|_| panic!("context provided in current scope"))
    }

    pub fn try_use_context<T>(self) -> Option<T::Output>
    where
        T: 'static + Store<'a>,
    {
        use_context_impl::<T>(self.inherited()).map(|source| T::make_output(self, source))
    }

    pub fn use_context<T>(self) -> T::Output
    where
        T: 'static + Store<'a>,
    {
        self.try_use_context::<T>().expect("context not provided")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::*;

    #[test]
    fn provide_and_use_context() {
        Scope::create_root(|cx| {
            cx.provide_context(ReactiveStore(777i32));
            let x = cx.use_context::<ReactiveStore<i32>>();
            assert_eq!(*x.get(), 777);
        });
    }

    #[test]
    fn use_context_from_child_scope() {
        Scope::create_root(|cx| {
            cx.provide_context(PlainStore(777i32));
            cx.create_child(|cx| {
                let x = cx.use_context::<PlainStore<i32>>();
                assert_eq!(*x, 777);
            });
        });
    }

    #[test]
    fn unique_context_in_same_scope() {
        Scope::create_root(|cx| {
            cx.provide_context(PlainStore(777i32));
            assert!(cx.try_provide_context(PlainStore(777i32)).is_err());
        });
    }
}
