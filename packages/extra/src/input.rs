#![allow(dead_code)]

use wasm_bindgen::{intern, JsCast};
use xframe_core::{Attribute, GenericElement, GenericNode, IntoEventHandler, IntoReactive};
use xframe_reactive::Scope;

pub(crate) type JsBoolean = bool;
pub(crate) type JsNumber = f64;
pub(crate) type JsString = Attribute;

type CowStr = std::borrow::Cow<'static, str>;

pub(crate) struct BaseElement<N> {
    node: N,
    cx: Scope,
}

#[allow(dead_code)]
impl<N: GenericNode> BaseElement<N> {
    pub fn create(tag: &'static str, cx: Scope) -> Self {
        Self {
            cx,
            node: N::create(intern(tag).into()),
        }
    }

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
        self.set_property(CowStr::Borrowed(intern(name)), val.into_reactive().cast());
    }

    pub fn set_property(&self, name: impl Into<CowStr>, val: impl IntoReactive<Attribute>) {
        let attr = val.into();
        let name = name.into();
        let node = self.node.clone();
        self.cx.create_effect(move || {
            node.set_property(name.clone(), attr.clone().into_value());
        });
    }

    pub fn listen_event<Ev>(&self, event: &'static str, handler: impl IntoEventHandler<Ev>)
    where
        Ev: 'static + JsCast,
        N: GenericNode<Event = web_sys::Event>,
    {
        self.node.listen_event(
            intern(event).into(),
            handler
                .into_event_handler()
                .cast_with(|ev| ev.unchecked_into()),
        );
    }

    pub fn add_class<T: Into<CowStr>>(&self, name: T) {
        self.node.add_class(name.into());
    }

    pub fn append_child<E>(&self, element: E)
    where
        E: GenericElement<Node = N>,
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
