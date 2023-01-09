use crate::{
    event::EventHandler,
    str::{RcStr, StringLike},
    template::ThreadLocalState,
};
use std::fmt::Debug;

#[derive(Clone)]
pub enum NodeType {
    Tag(RcStr),
    TagNs { tag: RcStr, ns: RcStr },
    Text(RcStr),
    Placeholder(RcStr),
    Template(RcStr),
}

pub trait GenericNode: 'static + Clone + Debug + Eq {
    type Event;

    fn global_state() -> ThreadLocalState<Self>;

    fn create(ty: NodeType) -> Self;
    fn deep_clone(&self) -> Self;

    fn set_inner_text(&self, data: RcStr);
    fn set_property(&self, name: RcStr, attr: StringLike);
    fn set_attribute(&self, name: RcStr, attr: StringLike);

    fn add_class(&self, name: RcStr);
    fn remove_class(&self, name: RcStr);

    fn listen_event(&self, event: RcStr, handler: EventHandler<Self::Event>);

    fn parent(&self) -> Option<Self>;
    fn first_child(&self) -> Option<Self>;
    fn next_sibling(&self) -> Option<Self>;

    fn append_child(&self, child: &Self);
    fn replace_child(&self, new_node: &Self, old_node: &Self);
    fn remove_child(&self, node: &Self);
    fn insert_before(&self, new_node: &Self, ref_node: Option<&Self>);

    fn empty_template() -> Self {
        Self::create(NodeType::Template("".into()))
    }
}
