mod input {
    #[cfg(feature = "extra-elements")]
    pub(crate) use crate::{
        attr::AsCowStr,
        element::{BaseElement, GenericElement},
        node::GenericNode,
    };

    #[cfg(feature = "extra-elements")]
    pub(crate) use web_sys;

    #[cfg(feature = "extra-attributes")]
    pub(crate) fn cow_str_from_literal(s: &'static str) -> std::borrow::Cow<str> {
        wasm_bindgen::intern(s).into()
    }

    #[cfg(feature = "extra-events")]
    pub(crate) use crate::event::EventHandler;
}

include!(concat!(env!("OUT_DIR"), "/web_types.rs"));
