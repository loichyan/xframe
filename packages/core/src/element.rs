use crate::node::GenericNode;
use xframe_reactive::Scope;

pub trait GenericElement: Sized {
    type Node: GenericNode;
    fn create(cx: Scope) -> Self;
    fn create_with_node(cx: Scope, node: Self::Node) -> Self;
    fn into_node(self) -> Self::Node;
}
