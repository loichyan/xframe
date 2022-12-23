#![allow(dead_code)]

use std::borrow::Cow;
use wasm_bindgen::JsCast;
use xframe_core::{Attribute, GenericElement, GenericNode, IntoEventHandler, IntoReactive};
use xframe_reactive::Scope;
use xframe_web::WebNode;

pub(crate) type JsBoolean = bool;
pub(crate) type JsNumber = f64;
pub(crate) type JsString = Attribute;

type CowStr = std::borrow::Cow<'static, str>;

#[derive(Clone)]
pub(crate) struct BaseElement<N> {
    cx: Scope,
    node: N,
}

#[allow(dead_code)]
impl<N: GenericNode> BaseElement<N> {
    pub fn create_with_node(cx: Scope, node: N) -> Self {
        Self { cx, node }
    }

    pub fn node(&self) -> &N {
        &self.node
    }

    pub fn into_node(self) -> N {
        self.node
    }

    pub fn as_web_sys_element<T>(&self) -> &T
    where
        N: AsRef<web_sys::Node>,
        T: JsCast,
    {
        self.node.as_ref().unchecked_ref()
    }

    pub fn set_property_literal<T>(&self, name: &'static str, val: impl IntoReactive<T>)
    where
        T: 'static + Into<Attribute>,
    {
        self.set_property(Cow::Borrowed(name), val.into_reactive().cast());
    }

    pub fn set_property(&self, name: impl Into<CowStr>, val: impl IntoReactive<Attribute>) {
        let attr = val.into();
        let name = name.into();
        let node = self.node.clone();
        self.cx.create_effect(move || {
            node.set_property(name.clone(), attr.clone().into_value());
        });
    }

    pub fn listen_event<Ev>(&self, event: impl Into<CowStr>, handler: impl IntoEventHandler<Ev>)
    where
        Ev: 'static + JsCast,
        N: WebNode,
    {
        self.node.listen_event(
            event.into(),
            handler
                .into_event_handler()
                .cast_with(|ev: web_sys::Event| ev.unchecked_into()),
        );
    }

    pub fn add_class(&self, name: impl Into<CowStr>) {
        self.node.add_class(name.into());
    }

    pub fn toggle_class(&self, name: impl Into<CowStr>, toggle: impl IntoReactive<bool>) {
        let name = name.into();
        let toggle = toggle.into_reactive();
        let node = self.node.clone();
        self.cx.create_effect(move || {
            if toggle.clone().into_value() {
                node.add_class(name.clone());
            } else {
                node.remove_class(name.clone());
            }
        });
    }

    pub fn append_child<E>(&self, element: E)
    where
        E: GenericElement<N>,
    {
        self.node.append_child(&element.into_node());
    }

    pub fn append_children<I>(&self, nodes: I)
    where
        I: IntoIterator<Item = N>,
    {
        for node in nodes {
            self.node.append_child(&node);
        }
    }
}
