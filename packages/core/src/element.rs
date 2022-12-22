use crate::node::{GenericNode, NodeType};
use xframe_reactive::Scope;

pub trait GenericElement<N: GenericNode>: 'static + Clone {
    const TYPE: NodeType;
    fn create_with_node(cx: Scope, node: N) -> Self;
    fn into_node(self) -> N;
    fn create(cx: Scope) -> Self {
        Self::create_with_node(cx, N::create(Self::TYPE))
    }
}
