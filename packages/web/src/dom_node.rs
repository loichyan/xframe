use crate::DOCUMENT;
use js_sys::Reflect;
use wasm_bindgen::{prelude::*, JsCast};
use web_sys::AddEventListenerOptions;
use xframe_core::{Attribute, EventHandler, GenericNode};

type Str = std::borrow::Cow<'static, str>;

#[derive(Clone)]
pub struct DomNode {
    node: web_sys::Node,
}

impl From<web_sys::Node> for DomNode {
    fn from(node: web_sys::Node) -> Self {
        Self { node }
    }
}

impl From<DomNode> for web_sys::Node {
    fn from(t: DomNode) -> Self {
        t.node
    }
}

impl AsRef<web_sys::Node> for DomNode {
    fn as_ref(&self) -> &web_sys::Node {
        &self.node
    }
}

impl GenericNode for DomNode {
    type Event = web_sys::Event;

    fn create(tag: Str) -> Self {
        Self {
            node: DOCUMENT
                .with(|document| document.create_element(&tag))
                .unwrap()
                .into(),
        }
    }

    fn create_text_node(data: &str) -> Self {
        Self {
            node: DOCUMENT.with(|docuement| docuement.create_text_node(data).into()),
        }
    }

    fn create_fragment() -> Self {
        Self {
            node: DOCUMENT.with(|docuemnt| docuemnt.create_document_fragment().into()),
        }
    }

    fn deep_clone(&self) -> Self {
        Self {
            node: self.node.clone_node_with_deep(true).unwrap(),
        }
    }

    fn set_inner_text(&self, data: &str) {
        self.node.set_text_content(Some(data));
    }

    fn set_property(&self, name: Str, attr: Attribute) {
        Reflect::set(&self.node, &JsValue::from_str(&name), &attr.into_js_value()).unwrap();
    }

    fn set_attribute(&self, name: Str, val: Attribute) {
        self.node
            .unchecked_ref::<web_sys::Element>()
            .set_attribute(&name, val.into_string_only().as_str())
            .unwrap();
    }

    fn add_class(&self, name: Str) {
        self.node
            .unchecked_ref::<web_sys::Element>()
            .class_list()
            .add_1(&name)
            .unwrap();
    }

    fn listen_event(&self, event: Str, handler: EventHandler<Self::Event>) {
        let mut options = AddEventListenerOptions::default();
        options.capture(handler.options.capture);
        options.once(handler.options.once);
        options.passive(handler.options.passive);
        self.node
            .add_event_listener_with_callback_and_add_event_listener_options(
                &event,
                &Closure::wrap(handler.handler)
                    .into_js_value()
                    .unchecked_into(),
                &options,
            )
            .unwrap();
    }

    fn append_child(&self, child: Self) {
        self.node.append_child(&child.node).unwrap();
    }

    fn first_child(&self) -> Option<Self> {
        self.node.first_child().map(Self::from)
    }

    fn next_sibling(&self) -> Option<Self> {
        self.node.next_sibling().map(Self::from)
    }
}
