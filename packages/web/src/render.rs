use crate::{dom_node::DomNode, DOCUMENT};
use xframe_core::GenericComponent;
use xframe_reactive::Scope;

pub fn mount_to_body<C>(f: impl FnOnce(Scope) -> C)
where
    C: GenericComponent<DomNode>,
{
    let body = DOCUMENT.with(|docuemt| docuemt.body().unwrap());
    mount_to(&body, f);
}

pub fn mount_to<C>(root: &web_sys::Node, f: impl FnOnce(Scope) -> C)
where
    C: GenericComponent<DomNode>,
{
    xframe_reactive::create_root(|cx| {
        f(cx).mount_to(&DomNode::from(root.clone()));
    })
    .1
    .leak();
}
