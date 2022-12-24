#![allow(clippy::type_complexity)]

mod attr;
mod element;
mod event;
mod ext;
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
        element::{GenericElement, NodeRef},
        event::{EventHandler, EventOptions, IntoEventHandler},
        ext::ScopeExt,
        node::{GenericNode, NodeType},
        reactive::{IntoReactive, Reactive, Value},
        template::{Template, TemplateId},
        view::View,
    };
}

type CowStr = std::borrow::Cow<'static, str>;

#[macro_export]
macro_rules! is_debug {
    () => {
        cfg!(debug_assertions)
    };
}
