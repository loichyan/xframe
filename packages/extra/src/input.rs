#![allow(dead_code)]

use std::borrow::Cow;
use wasm_bindgen::JsCast;
use xframe_core::{
    component::Element, Attribute, GenericComponent, GenericNode, IntoEventHandler, IntoReactive,
    NodeType, Reactive, RenderOutput,
};
use xframe_reactive::Scope;
use xframe_web::{elements::text, WebNode};

pub(crate) type JsBoolean = bool;
pub(crate) type JsNumber = f64;
pub(crate) type JsString = Attribute;

type CowStr = std::borrow::Cow<'static, str>;

pub(crate) struct BaseElement<N> {
    inner: Element<N>,
}

#[allow(dead_code)]
impl<N: GenericNode> BaseElement<N> {
    pub fn new(cx: Scope, ty: NodeType) -> Self {
        Self {
            inner: Element::new(cx, ty),
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
        T: 'static + Into<Attribute>,
    {
        self.set_property(Cow::Borrowed(name), val.into_reactive(self.inner.cx).cast());
    }

    pub fn set_property(&self, name: impl Into<CowStr>, val: impl IntoReactive<Attribute>) {
        let val = val.into_reactive(self.inner.cx);
        self.inner.set_property(name.into(), val);
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
        self.inner.set_class(name.into(), Reactive::Value(true));
    }

    pub fn classes<I: IntoIterator<Item = &'static str>>(&self, names: I) {
        for name in names {
            self.node().add_class(name.into());
        }
    }

    pub fn toggle_class(&self, name: impl Into<CowStr>, toggle: impl IntoReactive<bool>) {
        self.inner
            .set_class(name.into(), toggle.into_reactive(self.inner.cx));
    }

    pub fn child<C: GenericComponent<N>>(&mut self, child: impl 'static + FnOnce() -> C) {
        self.inner.add_child(child);
    }

    pub fn child_text<A: IntoReactive<Attribute>>(&mut self, data: A) {
        let cx = self.inner.cx;
        let data = data.into_reactive(cx);
        self.child(move || text(cx).data(data));
    }
}
