use std::any::Any;
use xframe_core::{component::Element, GenericComponent, GenericNode, NodeType, View};
use xframe_reactive::{Scope, Signal};

pub trait GenericElement<N: GenericNode>:
    'static + AsRef<Element<N>> + AsMut<Element<N>> + GenericComponent<N>
{
    const TYPE: NodeType;

    fn dyn_view(mut self, dyn_view: impl 'static + FnMut(View<N>) -> View<N>) -> Self {
        if self.as_ref().is_dyn_view() {
            panic!("`Element::dyn_view` has been specified")
        }
        self.as_mut().set_dyn_view(dyn_view);
        self
    }

    fn child<C: GenericComponent<N>>(mut self, child: impl 'static + FnOnce() -> C) -> Self {
        self.as_mut().add_child(child);
        self
    }

    fn ref_(self, ref_: NodeRef<N>) -> Self {
        ref_.inner.set(Some(self.as_ref().root().clone()));
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
