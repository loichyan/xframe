#![allow(clippy::all)]

#[allow(unused_imports)]
mod input {
    pub(crate) use crate::{
        attr::{Attribute, IntoAttribute},
        element::{BaseElement, GenericElement},
        event::EventHandler,
        node::GenericNode,
    };
    pub(crate) use ::web_sys;
    pub(crate) use ::xframe;
}

#[cfg(feature = "extra-elements")]
include!(concat!(env!("OUT_DIR"), "/web_types.rs"));
