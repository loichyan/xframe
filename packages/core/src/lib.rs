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
        str::{RcStr, StringLike},
        template::TemplateId,
        view::View,
    };
}

pub type RandomState = std::hash::BuildHasherDefault<rustc_hash::FxHasher>;
pub type HashMap<K, V> = std::collections::HashMap<K, V, RandomState>;
pub type HashSet<T> = std::collections::HashSet<T, RandomState>;
