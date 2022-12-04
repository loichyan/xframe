mod generated;
mod node;

pub mod attr;
pub mod element;
pub mod event;

#[doc(inline)]
pub use self::{
    attr::AsCowStr,
    element::GenericElement,
    event::{EventHandler, EventHandlerWithOptions},
    node::{DomNode, GenericNode},
};

type Str<'a> = std::borrow::Cow<'a, str>;
