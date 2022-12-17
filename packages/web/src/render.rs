use crate::{dom_node::DomNode, DOCUMENT};
use xframe_core::GenericComponent;
use xframe_reactive::Scope;

pub fn render_to_body<C>(f: impl FnOnce(Scope) -> C)
where
    C: GenericComponent<DomNode>,
{
    let body = DOCUMENT.with(|docuemt| docuemt.body().unwrap());
    render(&body, f);
}

pub fn render<C>(root: &web_sys::Node, f: impl FnOnce(Scope) -> C)
where
    C: GenericComponent<DomNode>,
{
    xframe_reactive::create_root(|cx| {
        f(cx).render_to(&DomNode::from(root.clone()));
    })
    .leak();
}
