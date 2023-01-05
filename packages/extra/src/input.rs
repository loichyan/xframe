#![allow(dead_code)]

use std::borrow::Cow;
use wasm_bindgen::JsCast;
use xframe_core::{
    component::Element, GenericNode, IntoEventHandler, IntoReactive, NodeType, Reactive,
    RenderOutput, StringLike,
};
use xframe_reactive::Scope;
use xframe_web::WebNode;

pub(crate) type JsBoolean = bool;
pub(crate) type JsNumber = f64;
pub(crate) type JsString = StringLike;

type CowStr = std::borrow::Cow<'static, str>;

pub(crate) struct BaseElement<N> {
    inner: Element<N>,
}

#[allow(dead_code)]
impl<N: GenericNode> BaseElement<N> {
    pub fn new(cx: Scope, ty: NodeType) -> Self {
        Self {
            inner: Element::new(cx, || N::create(ty.clone())),
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
        self.set_property(Cow::Borrowed(name), val.into_reactive().cast());
    }

    pub fn set_property(&self, name: impl Into<CowStr>, val: impl IntoReactive<StringLike>) {
        let val = val.into_reactive();
        self.inner.set_property(name.into(), val);
    }

    pub fn set_attribute(&self, name: impl Into<CowStr>, val: impl IntoReactive<StringLike>) {
        let val = val.into_reactive();
        self.inner.set_attribute(name.into(), val);
    }

    pub fn listen_event<Ev>(&self, event: impl Into<CowStr>, handler: impl IntoEventHandler<Ev>)
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

    pub fn add_class(&self, name: impl Into<CowStr>) {
        self.inner.set_class(name.into(), Reactive::Static(true));
    }

    pub fn classes<I: IntoIterator<Item = &'static str>>(&self, names: I) {
        for name in names {
            self.inner.set_class(name.into(), Reactive::Static(true));
        }
    }

    pub fn toggle_class(&self, name: impl Into<CowStr>, toggle: impl IntoReactive<bool>) {
        self.inner.set_class(name.into(), toggle.into_reactive());
    }
}
