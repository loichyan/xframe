use std::borrow::Cow;

use wasm_bindgen::{intern, JsCast};
use xframe_core::{Attribute, GenericElement, GenericNode, IntoEventHandler, Reactive};
use xframe_reactive::Scope;

pub(crate) struct BaseElement<N> {
    node: N,
    cx: Scope,
}

impl<N: GenericNode> GenericElement for BaseElement<N> {
    type Node = N;
    fn into_node(self) -> Self::Node {
        self.node
    }
}

#[allow(dead_code)]
impl<N: GenericNode> BaseElement<N> {
    pub fn create(tag: &'static str, cx: Scope) -> Self {
        Self {
            node: N::create(intern(tag).into()),
            cx,
        }
    }

    pub fn node(&self) -> &N {
        &self.node
    }

    pub fn as_web_sys_element<T>(&self) -> &T
    where
        N: AsRef<web_sys::Node>,
        T: JsCast,
    {
        self.node.as_ref().unchecked_ref()
    }

    pub fn set_property_literal(&self, name: &'static str, val: impl Into<Reactive<Attribute>>) {
        self.set_property(intern(name).into(), val.into());
    }

    pub fn set_property(&self, name: Cow<'static, str>, val: impl Into<Reactive<Attribute>>) {
        let attr = val.into();
        let node = self.node.clone();
        self.cx.create_effect(move |_| {
            node.set_property(name.clone(), attr.clone().into_value());
        });
    }

    pub fn listen_event<Ev>(&self, event: &'static str, handler: impl IntoEventHandler<Ev>)
    where
        Ev: 'static + JsCast,
        N: GenericNode<Event = web_sys::Event>,
    {
        self.node
            .listen_event(intern(event).into(), handler.into_event_handler().cast());
    }
}
