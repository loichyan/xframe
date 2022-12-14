mod attr;
mod element;
mod event;
mod node;
mod reactive;

pub mod component;

#[doc(inline)]
pub use {
    attr::Attribute,
    component::GenericComponent,
    element::GenericElement,
    event::{EventHandler, EventOptions, IntoEventHandler},
    node::GenericNode,
    reactive::{IntoReactive, Reactive, Value},
};

type Str = std::borrow::Cow<'static, str>;

pub trait UnwrapThrowValExt<T> {
    fn unwrap_throw_val(self) -> T;
}

impl<T> UnwrapThrowValExt<T> for Result<T, wasm_bindgen::JsValue> {
    fn unwrap_throw_val(self) -> T {
        self.unwrap_or_else(|e| wasm_bindgen::throw_val(e))
    }
}
