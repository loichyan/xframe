mod attr;
mod element;
mod event;
mod node;

#[doc(inline)]
pub use self::{
    attr::{Attribute, IntoAttribute},
    element::GenericElement,
    event::{EventHandler, IntoEventHandler},
    node::GenericNode,
};

type Str = std::borrow::Cow<'static, str>;
