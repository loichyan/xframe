use crate::{dom_node::DomNode, Root, DOCUMENT};
use xframe_core::{GenericComponent, GenericNode};
use xframe_reactive::Scope;

pub fn mount_to_body<C: GenericComponent<DomNode>>(f: impl 'static + FnOnce(Scope) -> C) {
    let body = DOCUMENT.with(|docuemt| docuemt.body().unwrap());
    mount_to(&body, f);
}

pub fn mount_to<C: GenericComponent<DomNode>>(
    root: &web_sys::Node,
    f: impl 'static + FnOnce(Scope) -> C,
) {
    let (_, dispoer) = xframe_reactive::create_root(|cx| {
        Root(cx)
            .child(move || f(cx))
            .render_view()
            .append_to(&DomNode::from(root.clone()));
    });
    dispoer.leak();
}

/// A trait alias of [`xframe_core::GenericNode`].
pub trait WebNode: xframe_core::GenericNode<Event = web_sys::Event> {}

impl<T: GenericNode<Event = web_sys::Event>> WebNode for T {}
