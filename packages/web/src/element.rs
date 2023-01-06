use crate::{CowStr, GenericChild};
use std::any::Any;
use xframe_core::{
    component::Element, GenericComponent, GenericNode, IntoEventHandler, IntoReactive, NodeType,
    Reactive, StringLike,
};
use xframe_reactive::{Scope, Signal};

pub trait GenericElement<N: GenericNode>:
    'static + AsRef<Element<N>> + AsMut<Element<N>> + GenericComponent<N>
{
    const TYPE: NodeType;

    fn child(mut self, child: impl GenericChild<N>) -> Self {
        let cx = self.as_ref().cx;
        self.as_mut().add_child(move || child.render(cx));
        self
    }

    fn ref_(self, ref_: NodeRef<N>) -> Self {
        ref_.inner.set(Some(self.as_ref().root().clone()));
        self
    }

    fn attr<K, V>(self, name: K, val: V) -> Self
    where
        K: Into<CowStr>,
        V: IntoReactive<StringLike>,
    {
        self.as_ref()
            .set_attribute(name.into(), val.into_reactive());
        self
    }

    fn prop<K, V>(self, name: K, val: V) -> Self
    where
        K: Into<CowStr>,
        V: IntoReactive<StringLike>,
    {
        self.as_ref().set_property(name.into(), val.into_reactive());
        self
    }

    fn class<K, V>(self, class: K, toggle: V) -> Self
    where
        K: Into<CowStr>,
        V: IntoReactive<bool>,
    {
        self.as_ref()
            .set_class(class.into(), toggle.into_reactive());
        self
    }

    fn classes<I>(self, classes: I) -> Self
    where
        I: IntoIterator<Item = &'static str>,
    {
        for cls in classes {
            self.as_ref().set_class(cls.into(), Reactive::Static(true));
        }
        self
    }

    fn on<K, E>(self, event: K, handler: E) -> Self
    where
        K: Into<CowStr>,
        E: IntoEventHandler<N::Event>,
    {
        self.as_ref()
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
