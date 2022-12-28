use crate::element::NodeRef;
use std::borrow::Borrow;
use xframe_core::GenericNode;
use xframe_reactive::Scope;

pub trait ScopeExt: Borrow<Scope> {
    fn create_node_ref<N: GenericNode>(&self) -> NodeRef<N> {
        NodeRef::new(*self.borrow())
    }
}

impl<T: Borrow<Scope>> ScopeExt for T {}
