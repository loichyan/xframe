mod input {
    pub(super) use {
        crate::{
            attr::{cow_str_from_literal, AsCowStr},
            element::{BaseElement, GenericElement},
            event::EventHandler,
            node::GenericNode,
        },
        ::web_sys,
    };
}

include!(concat!(env!("OUT_DIR"), "/web_types.rs"));
