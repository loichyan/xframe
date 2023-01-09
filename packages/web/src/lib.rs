#![allow(clippy::type_complexity)]

#[macro_use]
mod utils;
mod child;
mod dom_node;
mod element;
mod event_delegation;
mod ext;
mod render;

pub mod components {
    #[path = "for.rs"]
    mod for_;
    mod fragment;
    mod list;
    mod root;
    mod switch;

    #[doc(inline)]
    pub use {
        for_::For,
        fragment::Fragment,
        list::List,
        root::Root,
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
        crate::{
            child::GenericChild,
            components::*,
            dom_node::DomNode,
            element::{GenericElement, NodeRef},
            ext::ScopeExt,
            render::*,
        },
        // TODO: wrap with XEvent
        web_sys::Event,
    };
}

#[doc(inline)]
pub use prelude::*;

thread_local! {
    static WINDOW: web_sys::Window = web_sys::window().unwrap();
    static DOCUMENT: web_sys::Document = WINDOW.with(web_sys::Window::document).unwrap();
}
