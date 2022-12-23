#![allow(clippy::type_complexity)]

#[macro_use]
mod utils;
mod dom_node;
mod event_delegation;
mod render;

pub mod components {
    pub(crate) mod element;
    #[path = "for.rs"]
    mod for_;
    mod fragment;
    mod list;
    mod switch;

    #[doc(inline)]
    pub use {
        element::Element,
        for_::For,
        fragment::Fragment,
        list::List,
        switch::{Else, If, Switch},
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
        crate::{components::element::view, components::*, dom_node::DomNode, render::*},
        web_sys::Event,
    };
}

#[doc(inline)]
pub use prelude::*;

thread_local! {
    static WINDOW: web_sys::Window = web_sys::window().unwrap();
    static DOCUMENT: web_sys::Document = WINDOW.with(web_sys::Window::document).unwrap();
}
