#![allow(dead_code)]

use std::borrow::Cow;
use wasm_bindgen::JsCast;
use xframe_core::{
    component::Element, Attribute, GenericComponent, GenericNode, IntoEventHandler, IntoReactive,
    NodeType, RenderInput, RenderOutput,
};
use xframe_web::WebNode;

pub(crate) type JsBoolean = bool;
pub(crate) type JsNumber = f64;
pub(crate) type JsString = Attribute;

type CowStr = std::borrow::Cow<'static, str>;

pub(crate) struct BaseElement<N> {
    inner: Element<N>,
}

#[allow(dead_code)]
impl<N: GenericNode> BaseElement<N> {
    pub fn new_with_input(input: RenderInput<N>, ty: NodeType) -> Self {
        Self {
            inner: Element::new_with_input(input, ty),
        }
    }

    pub fn render_to_output(self) -> RenderOutput<N> {
        self.inner.render_to_output()
    }

    pub fn node(&self) -> &N {
        self.inner.root()
    }

    pub fn as_element(&self) -> &Element<N> {
        &self.inner
    }

    pub fn as_element_mut(&mut self) -> &mut Element<N> {
        &mut self.inner
    }

    pub fn as_web_sys_element<T>(&self) -> &T
    where
        N: AsRef<web_sys::Node>,
        T: JsCast,
    {
        self.node().as_ref().unchecked_ref()
    }

    pub fn set_property_literal<T>(&self, name: &'static str, val: impl IntoReactive<T>)
    where
        T: 'static + Into<Attribute>,
    {
        self.set_property(Cow::Borrowed(name), val.into_reactive(self.inner.cx).cast());
    }

    pub fn set_property(&self, name: impl Into<CowStr>, val: impl IntoReactive<Attribute>) {
        let node = self.node().clone();
        let name = name.into();
        let attr = val.into_reactive(self.inner.cx);
        self.inner.cx.create_effect(move || {
            node.set_property(name.clone(), attr.clone().into_value());
        });
    }

    pub fn listen_event<Ev>(&self, event: impl Into<CowStr>, handler: impl IntoEventHandler<Ev>)
    where
        Ev: 'static + JsCast,
        N: WebNode,
    {
        self.node().listen_event(
            event.into(),
            handler
                .into_event_handler()
                .cast_with(|ev: web_sys::Event| ev.unchecked_into()),
        );
    }

    // TODO: pub fn classes(...)

    pub fn add_class(&self, name: impl Into<CowStr>) {
        self.node().add_class(name.into());
    }

    pub fn toggle_class(&self, name: impl Into<CowStr>, toggle: impl IntoReactive<bool>) {
        let node = self.node().clone();
        let name = name.into();
        let toggle = toggle.into_reactive(self.inner.cx);
        self.inner.cx.create_effect(move || {
            if toggle.clone().into_value() {
                node.add_class(name.clone());
            } else {
                node.remove_class(name.clone());
            }
        });
    }

    pub fn add_child<C: GenericComponent<N>>(&mut self, child: impl 'static + FnOnce(C) -> C) {
        self.inner.add_child(child);
    }

    pub fn append_children<I>(&self, nodes: I)
    where
        I: IntoIterator<Item = N>,
    {
        for node in nodes {
            self.node().append_child(&node);
        }
    }
}
