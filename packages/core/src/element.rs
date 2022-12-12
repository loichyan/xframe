use crate::node::GenericNode;

pub trait GenericElement: Sized {
    type Node: GenericNode;
    fn into_node(self) -> Self::Node;
}
