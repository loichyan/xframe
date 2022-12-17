use crate::{
    attr::Attribute,
    event::EventHandler,
    template::{TemplateId, Templates},
    CowStr,
};

pub enum NodeType {
    Tag(CowStr),
    Text(CowStr),
    Placeholder(CowStr),
    Template(Option<TemplateId>),
}

pub trait GenericNode: 'static + Clone + Eq {
    type Event;

    fn global_templates() -> Templates<Self>;
    fn create(ty: NodeType) -> Self;
    fn deep_clone(&self) -> Self;
    fn set_inner_text(&self, data: &str);
    fn set_property(&self, name: CowStr, attr: Attribute);
    fn set_attribute(&self, name: CowStr, attr: Attribute);
    fn add_class(&self, name: CowStr);
    fn listen_event(&self, event: CowStr, handler: EventHandler<Self::Event>);
    fn append_child(&self, child: &Self);
    fn first_child(&self) -> Option<Self>;
    fn next_sibling(&self) -> Option<Self>;
    fn parent(&self) -> Option<Self>;
    fn replace_child(&self, new_node: &Self, old_node: &Self);
    fn remove_child(&self, node: &Self);
    fn insert_before(&self, new_node: &Self, ref_node: Option<&Self>);
}
