mod attr;
mod component;
mod element;
mod event;
mod node;
mod reactive;

#[doc(inline)]
pub use self::{
    attr::Attribute,
    component::{create_component, Component, GenericComponent},
    element::GenericElement,
    event::{EventHandler, EventOptions, IntoEventHandler},
    node::GenericNode,
    reactive::{IntoReactive, Reactive, Value},
};

type Str = std::borrow::Cow<'static, str>;
