use crate::{attr::Attribute, event::EventHandler, Str};

pub trait GenericNode: 'static + Clone {
    type Event;

    fn create(tag: Str) -> Self;
    fn create_text_node(data: &str) -> Self;
    fn create_fragment() -> Self;
    fn deep_clone(&self) -> Self;
    fn set_inner_text(&self, data: &str);
    fn set_property(&self, name: Str, attr: Attribute);
    fn set_attribute(&self, name: Str, attr: Attribute);
    fn add_class(&self, name: Str);
    fn listen_event(&self, event: Str, handler: EventHandler<Self::Event>);
    fn append_child(&self, child: Self);
    fn first_child(&self) -> Option<Self>;
    fn next_sibling(&self) -> Option<Self>;
}
