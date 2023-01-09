use crate::{utils::UnwrapThrowValExt, DOCUMENT};
use js_sys::Reflect;
use std::borrow::Borrow;
use wasm_bindgen::{intern, prelude::*, JsCast};
use web_sys::HtmlTemplateElement;
use xframe_core::{
    is_debug,
    template::{GlobalState, ThreadLocalState},
    EventHandler, GenericNode, NodeType, RcStr, StringLike,
};

trait RcStrExt: Borrow<RcStr> {
    fn intern(&self) -> &str {
        match self.borrow() {
            RcStr::Literal(s) => intern(s),
            RcStr::Rc(s) => s,
        }
    }
}

impl<T: Borrow<RcStr>> RcStrExt for T {}

trait AttrExt: Into<StringLike> {
    fn into_js_value(self) -> JsValue {
        match self.into() {
            StringLike::Boolean(t) => JsValue::from_bool(t),
            StringLike::Integer(t) => JsValue::from_f64(t as f64),
            StringLike::Number(t) => JsValue::from_f64(t),
            StringLike::Literal(t) => JsValue::from_str(t),
            StringLike::String(t) => JsValue::from_str(&t),
        }
    }
}

impl<T: Into<StringLike>> AttrExt for T {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DomNode {
    node: web_sys::Node,
}

impl From<web_sys::Node> for DomNode {
    fn from(node: web_sys::Node) -> Self {
        Self::from_web_sys(node)
    }
}

impl From<DomNode> for web_sys::Node {
    fn from(t: DomNode) -> Self {
        t.into_web_sys()
    }
}

impl GenericNode for DomNode {
    type Event = web_sys::Event;

    fn global_state() -> ThreadLocalState<Self> {
        thread_local! {
            static STATE: GlobalState<DomNode> = GlobalState::default();
        }
        &STATE
    }

    fn create(ty: NodeType) -> Self {
        let node: web_sys::Node = DOCUMENT.with(|doc| match ty {
            NodeType::Tag(tag) => doc.create_element(tag.intern()).unwrap_throw_val().into(),
            NodeType::TagNs { tag, ns } => doc
                .create_element_ns(Some(ns.intern()), tag.intern())
                .unwrap_throw_val()
                .into(),
            NodeType::Text(data) => doc.create_text_node(data.intern()).into(),
            NodeType::Placeholder(desc) => doc.create_comment(desc.intern()).into(),
            NodeType::Template(data) => {
                if is_debug!() && !data.is_empty() {
                    let template = doc.create_element("template").unwrap_throw_val();
                    template
                        .set_attribute("data-xframe-template-id", data.intern())
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
        Self { node }
    }

    fn deep_clone(&self) -> Self {
        Self {
            node: self.node.clone_node_with_deep(true).unwrap_throw_val(),
        }
    }

    fn set_inner_text(&self, data: RcStr) {
        self.node.set_text_content(Some(&data));
    }

    fn set_property(&self, name: RcStr, attr: StringLike) {
        Reflect::set(
            &self.node,
            &JsValue::from_str(name.intern()),
            &attr.into_js_value(),
        )
        .unwrap_throw_val();
    }

    fn set_attribute(&self, name: RcStr, val: StringLike) {
        self.node
            .unchecked_ref::<web_sys::Element>()
            .set_attribute(name.intern(), val.into_string().intern())
            .unwrap_throw_val();
    }

    fn add_class(&self, name: RcStr) {
        self.node
            .unchecked_ref::<web_sys::Element>()
            .class_list()
            .add_1(name.intern())
            .unwrap_throw_val();
    }

    fn remove_class(&self, name: RcStr) {
        self.node
            .unchecked_ref::<web_sys::Element>()
            .class_list()
            .remove_1(name.intern())
            .unwrap_throw_val();
    }

    fn listen_event(&self, event: RcStr, handler: EventHandler<Self::Event>) {
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
    pub fn from_web_sys(node: web_sys::Node) -> Self {
        Self { node }
    }

    pub fn into_web_sys(self) -> web_sys::Node {
        self.node
    }

    pub fn as_web_sys(&self) -> &web_sys::Node {
        &self.node
    }
}
