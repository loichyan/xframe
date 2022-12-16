use std::borrow::Cow;

use crate::DOCUMENT;
use js_sys::Reflect;
use wasm_bindgen::{intern, prelude::*, JsCast};
use web_sys::AddEventListenerOptions;
use xframe_core::{
    component::Templates, Attribute, EventHandler, GenericNode, NodeType, UnwrapThrowValExt,
};

type CowStr = std::borrow::Cow<'static, str>;

trait CowStrExt {
    fn intern(&self) -> &str;
}

impl CowStrExt for CowStr {
    fn intern(&self) -> &str {
        match self {
            Cow::Borrowed(s) => intern(s),
            Cow::Owned(s) => s,
        }
    }
}

// TODO: use global IDs to check euqality
#[derive(Clone, Eq, PartialEq)]
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

thread_local! {
    static TEMPLATES: Templates<DomNode> = Templates::default();
}

impl GenericNode for DomNode {
    type Event = web_sys::Event;

    fn global_templates() -> Templates<Self> {
        TEMPLATES.with(Clone::clone)
    }

    fn create(ty: NodeType) -> Self {
        let node: web_sys::Node = DOCUMENT.with(|doc| match ty {
            NodeType::Tag(tag) => doc.create_element(tag.intern()).unwrap_throw_val().into(),
            NodeType::Text => doc.create_text_node("").into(),
            NodeType::Placeholder => doc.create_comment("").into(),
            NodeType::Template => doc.create_document_fragment().into(),
        });
        Self { node }
    }

    fn deep_clone(&self) -> Self {
        Self {
            node: self.node.clone_node_with_deep(true).unwrap_throw_val(),
        }
    }

    fn set_inner_text(&self, data: &str) {
        self.node.set_text_content(Some(data));
    }

    fn set_property(&self, name: CowStr, attr: Attribute) {
        Reflect::set(
            &self.node,
            &JsValue::from_str(name.intern()),
            &attr.into_js_value(),
        )
        .unwrap_throw_val();
    }

    fn set_attribute(&self, name: CowStr, val: Attribute) {
        self.node
            .unchecked_ref::<web_sys::Element>()
            .set_attribute(name.intern(), val.into_string_only().as_str())
            .unwrap_throw_val();
    }

    fn add_class(&self, name: CowStr) {
        self.node
            .unchecked_ref::<web_sys::Element>()
            .class_list()
            .add_1(name.intern())
            .unwrap_throw_val();
    }

    fn listen_event(&self, event: CowStr, handler: EventHandler<Self::Event>) {
        let mut options = AddEventListenerOptions::default();
        options.capture(handler.options.capture);
        options.once(handler.options.once);
        options.passive(handler.options.passive);
        self.node
            .add_event_listener_with_callback_and_add_event_listener_options(
                event.intern(),
                &Closure::wrap(handler.handler)
                    .into_js_value()
                    .unchecked_into(),
                &options,
            )
            .unwrap_throw_val();
    }

    fn append_child(&self, child: &Self) {
        self.node.append_child(&child.node).unwrap_throw_val();
    }

    fn first_child(&self) -> Option<Self> {
        self.node.first_child().map(Self::from)
    }

    fn next_sibling(&self) -> Option<Self> {
        self.node.next_sibling().map(Self::from)
    }

    fn parent(&self) -> Option<Self> {
        self.node.parent_node().map(Self::from)
    }

    fn replace_child(&self, new_node: &Self, old_node: &Self) {
        self.node
            .replace_child(&new_node.node, &old_node.node)
            .unwrap_throw_val();
    }

    fn remove_child(&self, node: &Self) {
        self.node.remove_child(&node.node).unwrap_throw_val();
    }

    fn insert_before(&self, new_node: &Self, ref_node: &Self) {
        self.node
            .insert_before(&new_node.node, Some(&ref_node.node))
            .unwrap_throw_val();
    }
}
