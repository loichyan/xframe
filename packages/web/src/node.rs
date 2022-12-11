use crate::{attr::Attribute, event::EventHandlerWithOptions, Str, DOCUMENT};
use js_sys::Reflect;
use wasm_bindgen::{prelude::*, JsCast};

pub trait GenericNode: 'static + Clone {
    fn create(tag: Str) -> Self;
    fn create_text_node(data: &str) -> Self;
    fn set_inner_text(&self, data: &str);
    fn set_property(&self, name: Str, attr: Attribute);
    fn set_attribute(&self, name: Str, attr: Attribute);
    fn add_class(&self, name: Str);
    fn listen_event(&self, event: Str, handler: EventHandlerWithOptions);
    fn append_child(&self, child: Self);
}

#[derive(Clone)]
pub struct DomNode {
    node: web_sys::Node,
}

impl AsRef<web_sys::Node> for DomNode {
    fn as_ref(&self) -> &web_sys::Node {
        &self.node
    }
}

impl GenericNode for DomNode {
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

    fn set_inner_text(&self, data: &str) {
        self.node.set_text_content(Some(data));
    }

    fn set_property(&self, name: Str, attr: Attribute) {
        Reflect::set(&self.node, &JsValue::from_str(&name), &attr.to_js_value()).unwrap();
    }

    fn set_attribute(&self, name: Str, val: Attribute) {
        self.node
            .unchecked_ref::<web_sys::Element>()
            .set_attribute(&name, val.to_string().as_str())
            .unwrap();
    }

    fn add_class(&self, name: Str) {
        self.node
            .unchecked_ref::<web_sys::Element>()
            .class_list()
            .add_1(&name)
            .unwrap();
    }

    fn listen_event(&self, event: Str, handler: EventHandlerWithOptions) {
        self.node
            .add_event_listener_with_callback_and_add_event_listener_options(
                &event,
                &Closure::wrap(handler.handler)
                    .into_js_value()
                    .unchecked_into(),
                &handler.options,
            )
            .unwrap();
    }

    fn append_child(&self, child: Self) {
        self.node.append_child(&child.node).unwrap();
    }
}
