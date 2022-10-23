use crate::{scope::ScopeInherited, signal::RawSignal, utils::ByAddress, Scope, Signal};
use ahash::AHashMap;
use std::{
    any::{Any, TypeId},
    cell::RefCell,
};

#[derive(Debug, Default)]
pub(crate) struct Contexts<'a> {
    inner: RefCell<AHashMap<TypeId, ByAddress<'a, dyn Any>>>,
}

fn type_id<T: 'static>() -> TypeId {
    TypeId::of::<RawSignal<T>>()
}

fn use_context_impl<'a, T: 'static>(inherited: &'a ScopeInherited) -> Option<Signal<'a, T>> {
    if let Some(any) = inherited.contexts.inner.borrow().get(&type_id::<T>()) {
        Some(downcast_context(any.0))
    } else {
        inherited.parent.and_then(use_context_impl)
    }
}

fn downcast_context<T: 'static>(any: &dyn Any) -> Signal<T> {
    let raw = any
        .downcast_ref::<RawSignal<T>>()
        .unwrap_or_else(|| unreachable!());
    Signal::from_raw(raw)
}

impl<'a> Scope<'a> {
    pub fn try_provide_context<T: 'static>(self, t: T) -> Result<Signal<'a, T>, Signal<'a, T>> {
        let signal = self.create_signal(t);
        let raw = signal.into_raw();
        if let Some(prev) = self
            .inherited()
            .contexts
            .inner
            .borrow_mut()
            .insert(type_id::<T>(), ByAddress(raw as &dyn Any))
        {
            Err(downcast_context(prev.0))
        } else {
            Ok(signal)
        }
    }

    pub fn provide_context<T: 'static>(self, t: T) -> Signal<'a, T> {
        self.try_provide_context(t)
            .unwrap_or_else(|_| panic!("context provided in current scope"))
    }

    pub fn try_use_context<T: 'static>(self) -> Option<Signal<'a, T>> {
        use_context_impl(self.inherited())
    }

    pub fn use_context<T: 'static>(self) -> Signal<'a, T> {
        self.try_use_context().expect("context not provided")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provide_and_use_context() {
        Scope::create_root(|cx| {
            cx.provide_context(777i32);
            let x = cx.use_context::<i32>();
            assert_eq!(*x.get(), 777);
        });
    }

    #[test]
    fn use_context_from_child_scope() {
        Scope::create_root(|cx| {
            cx.provide_context(777i32);
            cx.create_child(|cx| {
                let x = cx.use_context::<i32>();
                assert_eq!(*x.get(), 777);
            });
        });
    }

    #[test]
    fn unique_context_in_same_scope() {
        Scope::create_root(|cx| {
            cx.provide_context(777i32);
            assert!(cx.try_provide_context(777i32).is_err());
        });
    }
}
