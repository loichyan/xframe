#![allow(clippy::type_complexity)]

#[macro_export]
macro_rules! is_debug {
    () => {
        cfg!(debug_assertions)
    };
}

mod event;
mod node;
mod reactive;
mod str;

pub mod component;
pub mod template;
pub mod view;

#[doc(inline)]
pub use prelude::*;

pub mod prelude {
    #[doc(inline)]
    pub use crate::{
        component::{GenericComponent, RenderOutput},
        event::{EventHandler, EventOptions, IntoEventHandler},
        node::{GenericNode, NodeType},
        reactive::{IntoReactive, IntoReactiveValue, Reactive},
        str::StringLike,
        template::TemplateId,
        view::View,
    };
}

type CowStr = std::borrow::Cow<'static, str>;
