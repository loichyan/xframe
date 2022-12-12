#[allow(unused_imports)]
mod input {
    pub(crate) use crate::input::*;
    pub(crate) use ::web_sys;
    pub(crate) use xframe_core as core;
    pub(crate) use xframe_reactive as reactive;
}

include!(concat!(env!("OUT_DIR"), "/xframe_extra.rs"));
