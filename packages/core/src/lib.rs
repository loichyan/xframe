mod attr;
mod element;
mod event;
mod node;
mod reactive;

#[doc(inline)]
pub use self::{attr::*, element::*, event::*, node::*, reactive::*};

type Str = std::borrow::Cow<'static, str>;
