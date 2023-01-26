use crate::GenericChild;
use std::{any::Any, borrow::BorrowMut};
use xframe_core::{
    component::Element, GenericComponent, GenericNode, IntoEventHandler, IntoReactive, NodeType,
    RcStr, StringLike,
};
use xframe_reactive::{Scope, Signal};

// TODO: rename to `ElementExt`
pub trait GenericElement<N: GenericNode>:
    'static + BorrowMut<Element<N>> + GenericComponent<N>
{
    const TYPE: NodeType;

    fn child(mut self, child: impl GenericChild<N>) -> Self {
        let cx = self.borrow().cx;
        self.borrow_mut().add_child(move || child.render(cx));
        self
    }

    fn ref_(self, ref_: NodeRef<N>) -> Self {
        ref_.inner.set(Some(self.borrow().root().clone()));
        self
    }

    fn attr<K, V>(self, name: K, val: V) -> Self
    where
        K: Into<RcStr>,
        V: IntoReactive<StringLike>,
    {
        self.borrow()
            .set_attribute(name.into(), val.into_reactive());
        self
    }

    fn prop<K, V>(self, name: K, val: V) -> Self
    where
        K: Into<RcStr>,
        V: IntoReactive<StringLike>,
    {
        self.borrow().set_property(name.into(), val.into_reactive());
        self
    }

    fn class<K, V>(self, class: K, toggle: V) -> Self
    where
        K: Into<RcStr>,
        V: IntoReactive<bool>,
    {
        self.borrow()
            .set_class(class.into(), toggle.into_reactive());
        self
    }

    fn classes<I>(self, classes: I) -> Self
    where
        I: AsRef<[&'static str]>,
    {
        self.borrow().set_classes(classes.as_ref());
        self
    }

    fn on<K, E>(self, event: K, handler: E) -> Self
    where
        K: Into<RcStr>,
        E: IntoEventHandler<N::Event>,
    {
        self.borrow()
            .listen_event(event.into(), handler.into_event_handler());
        self
    }
}

pub struct NodeRef<N: 'static> {
    inner: Signal<Option<N>>,
}

impl<N> Clone for NodeRef<N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<N> Copy for NodeRef<N> {}

impl<N: GenericNode> NodeRef<N> {
    pub(crate) fn new(cx: Scope) -> Self {
        Self {
            inner: cx.create_signal(None),
        }
    }

    /// Get the reference and cast to the specified type.
    pub fn get_as<U: GenericNode>(&self) -> Option<U> {
        (&self.get() as &dyn Any).downcast_ref::<U>().cloned()
    }

    pub fn get(&self) -> N {
        self.try_get().expect("`NodeRef` has not been bound")
    }

    pub fn try_get(&self) -> Option<N> {
        self.inner.get()
    }
}
