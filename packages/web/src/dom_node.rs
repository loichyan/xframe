use crate::{utils::UnwrapThrowValExt, DOCUMENT};
use js_sys::Reflect;
use std::{
    borrow::{Borrow, Cow},
    cell::Cell,
};
use wasm_bindgen::{intern, prelude::*, JsCast};
use web_sys::HtmlTemplateElement;
use xframe_core::{is_debug, Attribute, EventHandler, GenericNode, NodeType};

thread_local! {
    static GLOBAL_ID: Cell<usize> = Cell::new(0);
}

type CowStr = std::borrow::Cow<'static, str>;

trait CowStrExt: Borrow<CowStr> {
    fn intern(&self) -> &str {
        match self.borrow() {
            Cow::Borrowed(s) => intern(s),
            Cow::Owned(s) => s,
        }
    }
}

impl<T: Borrow<CowStr>> CowStrExt for T {}

trait AttrExt: Into<Attribute> {
    fn into_js_value(self) -> JsValue {
        match self.into() {
            Attribute::Boolean(t) => JsValue::from_bool(t),
            Attribute::Number(t) => JsValue::from_f64(t),
            Attribute::Static(t) => JsValue::from_str(t),
            Attribute::String(t) => JsValue::from_str(&t),
        }
    }
}

impl<T: Into<Attribute>> AttrExt for T {}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct NodeId {
    id: usize,
}

impl NodeId {
    pub fn new() -> Self {
        GLOBAL_ID.with(|id| {
            let current = id.get();
            id.set(current + 1);
            NodeId { id: current }
        })
    }
}

#[derive(Clone)]
pub struct DomNode {
    id: NodeId,
    node: web_sys::Node,
}

impl PartialEq for DomNode {
    fn eq(&self, other: &Self) -> bool {
        self.id.eq(&other.id) || self.node.eq(&other.node)
    }
}

impl Eq for DomNode {}

impl From<web_sys::Node> for DomNode {
    fn from(node: web_sys::Node) -> Self {
        Self::from_js(node)
    }
}

impl From<DomNode> for web_sys::Node {
    fn from(t: DomNode) -> Self {
        t.into_js()
    }
}

impl AsRef<web_sys::Node> for DomNode {
    fn as_ref(&self) -> &web_sys::Node {
        self.as_js()
    }
}

impl GenericNode for DomNode {
    type Event = web_sys::Event;

    fn create(ty: NodeType) -> Self {
        let node: web_sys::Node = DOCUMENT.with(|doc| match ty {
            NodeType::Tag(tag) => doc.create_element(tag.intern()).unwrap_throw_val().into(),
            NodeType::Text(data) => doc.create_text_node(data.intern()).into(),
            NodeType::Placeholder(desc) => doc.create_comment(desc.intern()).into(),
            NodeType::Template(data) => {
                if is_debug!() && !data.is_empty() {
                    let template = doc.create_element("template").unwrap_throw_val();
                    template
                        .set_attribute("data-xframe-template-id", &data)
                        .unwrap_throw_val();
                    let body = doc.body().unwrap_throw();
                    body.insert_before(&template, body.first_child().as_ref())
                        .unwrap_throw_val();
                    template
                        .unchecked_into::<HtmlTemplateElement>()
                        .content()
                        .into()
                } else {
                    doc.create_document_fragment().into()
                }
            }
        });
        Self {
            node,
            id: NodeId::new(),
        }
    }

    fn deep_clone(&self) -> Self {
        Self {
            id: NodeId::new(),
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
            .set_attribute(name.intern(), val.into_string().intern())
            .unwrap_throw_val();
    }

    fn add_class(&self, name: CowStr) {
        self.node
            .unchecked_ref::<web_sys::Element>()
            .class_list()
            .add_1(name.intern())
            .unwrap_throw_val();
    }

    fn remove_class(&self, name: CowStr) {
        self.node
            .unchecked_ref::<web_sys::Element>()
            .class_list()
            .remove_1(name.intern())
            .unwrap_throw_val();
    }

    fn listen_event(&self, event: CowStr, handler: EventHandler<Self::Event>) {
        crate::event_delegation::add_event_listener(&self.node, event, handler);
    }

    fn parent(&self) -> Option<Self> {
        self.node.parent_node().map(Self::from)
    }

    fn first_child(&self) -> Option<Self> {
        self.node.first_child().map(Self::from)
    }

    fn next_sibling(&self) -> Option<Self> {
        self.node.next_sibling().map(Self::from)
    }

    fn append_child(&self, child: &Self) {
        self.node.append_child(&child.node).unwrap_throw_val();
    }

    fn replace_child(&self, new_node: &Self, old_node: &Self) {
        self.node
            .replace_child(&new_node.node, &old_node.node)
            .unwrap_throw_val();
    }

    fn remove_child(&self, node: &Self) {
        self.node.remove_child(&node.node).unwrap_throw_val();
    }

    fn insert_before(&self, new_node: &Self, ref_node: Option<&Self>) {
        self.node
            .insert_before(&new_node.node, ref_node.map(|node| &node.node))
            .unwrap_throw_val();
    }
}

impl DomNode {
    pub fn from_js(node: web_sys::Node) -> Self {
        Self {
            id: NodeId::new(),
            node,
        }
    }

    pub fn into_js(self) -> web_sys::Node {
        self.node
    }

    pub fn as_js(&self) -> &web_sys::Node {
        &self.node
    }
}
