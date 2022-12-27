use crate::{dom_node::DomNode, DOCUMENT};
use xframe_core::{GenericComponent, GenericNode};
use xframe_reactive::Scope;

pub fn mount_to_body<C: GenericComponent<DomNode>>(f: impl FnOnce(Scope) -> C) {
    let body = DOCUMENT.with(|docuemt| docuemt.body().unwrap());
    mount_to(&body, f);
}

pub fn mount_to<C: GenericComponent<DomNode>>(root: &web_sys::Node, f: impl FnOnce(Scope) -> C) {
    xframe_reactive::create_root(|cx| {
        f(cx).render().append_to(&DomNode::from(root.clone()));
    })
    .1
    .leak();
}

/// A trait alias of [`xframe_core::GenericNode`].
pub trait WebNode: xframe_core::GenericNode<Event = web_sys::Event> {}

impl<T: GenericNode<Event = web_sys::Event>> WebNode for T {}
