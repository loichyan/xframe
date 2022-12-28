#![allow(clippy::type_complexity)]

mod attr;
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
        component::{GenericComponent, RenderOutput},
        event::{EventHandler, EventOptions, IntoEventHandler},
        node::{GenericNode, NodeType},
        reactive::{IntoReactive, Reactive, Value},
        template::TemplateId,
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
