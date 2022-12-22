use std::any::Any;

use crate::node::{GenericNode, NodeType};
use xframe_reactive::{Scope, Signal};

pub trait GenericElement<N: GenericNode>: 'static + Clone {
    const TYPE: NodeType;
    fn create_with_node(cx: Scope, node: N) -> Self;
    fn into_node(self) -> N;
    fn create(cx: Scope) -> Self {
        Self::create_with_node(cx, N::create(Self::TYPE))
    }

    fn ref_(self, ref_: NodeRef<N>) -> Self {
        ref_.inner.set(Some(self.clone().into_node()));
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
