use crate::{attr::Attribute, event::EventHandlerWithOptions, Str};
use wasm_bindgen::{prelude::*, JsCast};
use web_sys::{Document, Window};

pub trait GenericNode {
    fn create(tag: Str) -> Self;
    fn set_attribute(&self, name: Str, attr: Attribute);
    fn add_class(&self, name: Str);
    fn listen_event(&self, event: Str, handler: EventHandlerWithOptions);
    fn append_child(&self, child: Self);
}

thread_local! {
    static WINDOW: Window = web_sys::window().unwrap();
    static DOCUMENT: Document = WINDOW.with(Window::document).unwrap();
}

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
                .unchecked_into(),
        }
    }

    fn set_attribute(&self, name: Str, val: Attribute) {
        val.read(|val| {
            self.node
                .unchecked_ref::<web_sys::Element>()
                .set_attribute(&name, val)
                .unwrap();
        });
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
