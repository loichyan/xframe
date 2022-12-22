#![allow(clippy::type_complexity)]

mod attr;
mod element;
mod event;
mod node;
mod reactive;

pub mod component;
pub mod template;
pub mod view;

#[doc(inline)]
pub use prelude::*;

pub mod prelude {
    #[doc(inline)]
    pub use crate::{
        attr::Attribute,
        component::GenericComponent,
        element::GenericElement,
        event::{EventHandler, EventOptions, IntoEventHandler},
        node::{GenericNode, NodeType},
        reactive::{IntoReactive, Reactive, Value},
        template::{Template, TemplateId},
        view::View,
    };
}

type CowStr = std::borrow::Cow<'static, str>;

#[macro_export]
macro_rules! is_dev {
    () => {
        cfg!(debug_assertions)
    };
}

// TODO: move to `xframe-web`
pub trait UnwrapThrowValExt<T> {
    fn unwrap_throw_val(self) -> T;
}

impl<T> UnwrapThrowValExt<T> for Result<T, wasm_bindgen::JsValue> {
    fn unwrap_throw_val(self) -> T {
        self.unwrap_or_else(|e| wasm_bindgen::throw_val(e))
    }
}
