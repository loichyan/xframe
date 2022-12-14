mod attr;
mod element;
mod event;
mod node;
mod reactive;

pub mod component;

#[doc(inline)]
pub use self::{
    attr::Attribute,
    component::GenericComponent,
    element::GenericElement,
    event::{EventHandler, EventOptions, IntoEventHandler},
    node::GenericNode,
    reactive::{IntoReactive, Reactive, Value},
};

type Str = std::borrow::Cow<'static, str>;
