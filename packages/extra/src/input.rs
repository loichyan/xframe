#![allow(dead_code)]

use wasm_bindgen::JsCast;
use xframe_core::{
    component::Element, GenericNode, IntoEventHandler, IntoReactive, RcStr, RenderOutput,
    StringLike,
};
use xframe_reactive::Scope;
use xframe_web::{GenericElement, WebNode};

pub mod attr_types {
    pub type BooleanValue = bool;
    pub type NumberValue = f64;
    pub type StringValue = xframe_core::StringLike;
}

pub struct ElementBase<N> {
    inner: Element<N>,
}

#[allow(dead_code)]
impl<N: GenericNode> ElementBase<N> {
    pub fn new<E: GenericElement<N>>(cx: Scope) -> Self {
        Self {
            inner: Element::new(cx, || N::create(E::TYPE)),
        }
    }

    pub fn render(self) -> RenderOutput<N> {
        self.inner.render()
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
        T: 'static + Into<StringLike>,
    {
        self.set_property(name, val.into_reactive().cast());
    }

    pub fn set_property(&self, name: impl Into<RcStr>, val: impl IntoReactive<StringLike>) {
        let val = val.into_reactive();
        self.inner.set_property(name.into(), val);
    }

    pub fn set_attribute_literal<T>(&self, name: &'static str, val: impl IntoReactive<T>)
    where
        T: 'static + Into<StringLike>,
    {
        self.set_attribute(name, val.into_reactive().cast());
    }

    pub fn set_attribute(&self, name: impl Into<RcStr>, val: impl IntoReactive<StringLike>) {
        let val = val.into_reactive();
        self.inner.set_attribute(name.into(), val);
    }

    pub fn listen_event<Ev>(&self, event: impl Into<RcStr>, handler: impl IntoEventHandler<Ev>)
    where
        Ev: 'static + JsCast,
        N: WebNode,
    {
        self.inner.listen_event(
            event.into(),
            handler
                .into_event_handler()
                .cast_with(|ev: web_sys::Event| ev.unchecked_into()),
        );
    }
}
