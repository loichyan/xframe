mod attr;
mod element;
mod event;
mod node;
mod reactive;

pub mod component;

#[doc(inline)]
pub use prelude::*;

pub mod prelude {
    #[doc(inline)]
    pub use crate::{
        attr::Attribute,
        component::{Component, GenericComponent, Template, TemplateId},
        element::GenericElement,
        event::{EventHandler, EventOptions, IntoEventHandler},
        node::{GenericNode, NodeType},
        reactive::{IntoReactive, Reactive, Value},
    };
}

type CowStr = std::borrow::Cow<'static, str>;

pub trait UnwrapThrowValExt<T> {
    fn unwrap_throw_val(self) -> T;
}

impl<T> UnwrapThrowValExt<T> for Result<T, wasm_bindgen::JsValue> {
    fn unwrap_throw_val(self) -> T {
        self.unwrap_or_else(|e| wasm_bindgen::throw_val(e))
    }
}
