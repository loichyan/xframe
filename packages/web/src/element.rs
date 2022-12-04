use crate::{attr::AsCowStr, event::EventHandler, GenericNode, Str};
use wasm_bindgen::{intern, JsCast};

pub use crate::generated::output::{element_types::*, elements::*};

pub mod prelude {
    pub use crate::generated::output::elements::*;
}

pub trait GenericElement: Sized {
    type Node: GenericNode;
    fn node(&self) -> &Self::Node;
    fn into_node(self) -> Self::Node;

    fn class(self, name: Str) -> Self {
        self.node().add_class(name);
        self
    }

    fn attribute(self, name: Str, attr: Str) -> Self {
        self.node().set_attribute(name, attr);
        self
    }

    fn event(self, event: Str, handler: impl EventHandler<web_sys::Event>) -> Self {
        self.node()
            .listen_event(event, handler.into_event_handler());
        self
    }

    fn child<E>(self, element: E) -> Self
    where
        E: GenericElement<Node = Self::Node>,
    {
        self.node().append_child(element.into_node());
        self
    }

    fn children<I>(self, nodes: I) -> Self
    where
        I: IntoIterator<Item = Self::Node>,
    {
        let node = self.node();
        for child in nodes {
            node.append_child(child);
        }
        self
    }
}

pub(crate) struct BaseElement<N>(N);

impl<N: GenericNode> GenericElement for BaseElement<N> {
    type Node = N;
    fn node(&self) -> &Self::Node {
        &self.0
    }
    fn into_node(self) -> Self::Node {
        self.0
    }
}

impl<N: GenericNode> BaseElement<N> {
    pub fn create(tag: &'static str) -> Self {
        Self(N::create(intern(tag).into()))
    }

    pub fn listen_event<Ev>(&self, event: &'static str, handler: impl EventHandler<Ev>)
    where
        Ev: 'static + JsCast,
    {
        self.0.listen_event(
            intern(event).into(),
            handler.into_event_handler().erase_type(),
        );
    }

    pub fn set_attribute<T>(&self, name: &'static str, val: T)
    where
        T: AsCowStr,
    {
        self.0.set_attribute(intern(name).into(), val.as_cow_str());
    }

    pub fn as_web_sys_element<T>(&self) -> &T
    where
        N: AsRef<web_sys::Node>,
        T: JsCast,
    {
        self.0.as_ref().unchecked_ref()
    }
}
