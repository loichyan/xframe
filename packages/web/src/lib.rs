mod dom_node;

pub mod elements {
    pub(crate) mod text;

    #[doc(inline)]
    pub use text::text;
}
pub mod element_types {
    #[doc(inline)]
    pub use crate::elements::text::Text;
}

#[doc(inline)]
pub use dom_node::DomNode;
use xframe_core::GenericElement;

thread_local! {
    static WINDOW: web_sys::Window = web_sys::window().unwrap();
    static DOCUMENT: web_sys::Document = WINDOW.with(web_sys::Window::document).unwrap();
}

pub fn render_to_body<E>(f: impl FnOnce(xframe_reactive::Scope) -> E)
where
    E: GenericElement<Node = DomNode>,
{
    let body = DOCUMENT.with(|docuemt| docuemt.body().unwrap());
    render(&body, f);
}

pub fn render<E>(root: &web_sys::Node, f: impl FnOnce(xframe_reactive::Scope) -> E)
where
    E: GenericElement<Node = DomNode>,
{
    xframe_reactive::create_root(|cx| {
        let node = f(cx).into_node();
        root.append_child(node.as_ref()).unwrap();
    })
    .leak();
}
