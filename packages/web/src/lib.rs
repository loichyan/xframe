#[macro_use]
mod macros;

mod generated;
mod node;

pub mod attr;
pub mod element;
pub mod event;

#[doc(inline)]
pub use xframe::*;

#[doc(inline)]
pub use self::{
    attr::{Attribute, IntoAttribute},
    element::GenericElement,
    event::{EventHandler, EventHandlerWithOptions},
    node::{DomNode, GenericNode},
};

thread_local! {
    static WINDOW: web_sys::Window = web_sys::window().unwrap();
    static DOCUMENT: web_sys::Document = WINDOW.with(web_sys::Window::document).unwrap();
}

type Str = std::borrow::Cow<'static, str>;

pub fn render_to_body<E>(f: impl FnOnce(Scope) -> E)
where
    E: GenericElement<Node = DomNode>,
{
    let body = DOCUMENT.with(|docuemt| docuemt.body().unwrap());
    render(&body, f);
}

pub fn render<E>(root: &web_sys::Node, f: impl FnOnce(Scope) -> E)
where
    E: GenericElement<Node = DomNode>,
{
    create_root(|cx| {
        let node = f(cx).into_node();
        root.append_child(node.as_ref()).unwrap();
    })
    .leak();
}
