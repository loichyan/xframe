mod component;
mod dom_node;

pub mod components {
    mod show;

    #[doc(inline)]
    pub use show::Show;
}

pub mod elements {
    #[path = "placeholder.rs"]
    mod placeholder_;
    #[path = "text.rs"]
    mod text_;

    #[doc(inline)]
    pub use {placeholder_::placeholder, text_::text};
}

pub mod element_types {}

#[doc(inline)]
pub use {
    ::web_sys::Event,
    component::{create_component, Component},
    dom_node::DomNode,
};

use xframe_core::GenericComponent;

thread_local! {
    static WINDOW: web_sys::Window = web_sys::window().unwrap();
    static DOCUMENT: web_sys::Document = WINDOW.with(web_sys::Window::document).unwrap();
}

pub fn render_to_body<C>(f: impl FnOnce(xframe_reactive::Scope) -> C)
where
    C: GenericComponent<Node = DomNode>,
{
    let body = DOCUMENT.with(|docuemt| docuemt.body().unwrap());
    render(&body, f);
}

pub fn render<C>(root: &web_sys::Node, f: impl FnOnce(xframe_reactive::Scope) -> C)
where
    C: GenericComponent<Node = DomNode>,
{
    xframe_reactive::create_root(|cx| {
        let node = f(cx).render();
        root.append_child(node.as_ref()).unwrap();
    })
    .leak();
}
