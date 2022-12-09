mod generated;
mod node;

pub mod attr;
pub mod element;
pub mod event;

#[doc(inline)]
pub use self::{
    attr::{Attribute, IntoAttribute},
    element::GenericElement,
    event::{EventHandler, EventHandlerWithOptions},
    node::{DomNode, GenericNode},
};

type Str = std::borrow::Cow<'static, str>;
