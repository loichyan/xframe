use crate::node::GenericNode;
use xframe_reactive::Scope;

pub trait GenericElement<N: GenericNode>: 'static + Sized {
    fn create(cx: Scope) -> Self;
    fn create_with_node(cx: Scope, node: N) -> Self;
    fn into_node(self) -> N;
}
