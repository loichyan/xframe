use std::fmt::Debug;

use crate::{attr::Attribute, event::EventHandler, template::ThreadLocalState, CowStr};

#[derive(Clone)]
pub enum NodeType {
    Tag(CowStr),
    Text(CowStr),
    Placeholder(CowStr),
    Template(CowStr),
}

pub trait GenericNode: 'static + Clone + Debug + Eq {
    type Event;

    fn global_state() -> ThreadLocalState<Self>;

    fn create(ty: NodeType) -> Self;
    fn deep_clone(&self) -> Self;

    fn set_inner_text(&self, data: &str);
    fn set_property(&self, name: CowStr, attr: Attribute);
    fn set_attribute(&self, name: CowStr, attr: Attribute);

    fn add_class(&self, name: CowStr);
    fn remove_class(&self, name: CowStr);

    fn listen_event(&self, event: CowStr, handler: EventHandler<Self::Event>);

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
