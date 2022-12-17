#![allow(clippy::type_complexity)]

#[macro_use]
mod utils;
mod dom_node;
mod render;

pub mod components {
    pub(crate) mod element;
    mod fragment;
    mod indexed;
    mod show;

    #[doc(inline)]
    pub use {
        element::Element,
        fragment::Fragment,
        indexed::Indexed,
        show::{Else, If, Show},
    };
}

pub mod elements {
    #[path = "text.rs"]
    mod text_;

    #[doc(inline)]
    pub use text_::text;
}

pub mod element_types {}

pub mod prelude {
    #[doc(inline)]
    pub use {
        crate::{
            components::element::{view, view_with},
            components::*,
            dom_node::DomNode,
            render::*,
        },
        web_sys::Event,
    };
}

#[doc(inline)]
pub use prelude::*;

thread_local! {
    static WINDOW: web_sys::Window = web_sys::window().unwrap();
    static DOCUMENT: web_sys::Document = WINDOW.with(web_sys::Window::document).unwrap();
}
