pub mod element {
    pub mod prelude {
        #[cfg(feature = "extra-elements")]
        #[doc(inline)]
        pub use xframe_extra::elements::*;
        #[doc(inline)]
        pub use xframe_web::elements::*;
    }
    #[doc(inline)]
    pub use prelude::*;

    #[cfg(feature = "extra-elements")]
    #[doc(inline)]
    pub use xframe_extra::element_types::*;
    #[doc(inline)]
    pub use xframe_web::element_types::*;
}

pub mod attr {
    #[cfg(feature = "extra-attributes")]
    #[doc(inline)]
    pub use xframe_extra::attr_types::*;
}

pub mod event {
    #[cfg(feature = "extra-events")]
    #[doc(inline)]
    pub use xframe_extra::event_types::*;
}

#[doc(inline)]
pub use xframe_core::{
    Attribute, EventOptions, GenericComponent, GenericElement, IntoEventHandler, IntoReactive,
    Reactive, Value,
};

/// A trait alias of [`xframe_core::GenericNode`].
pub trait GenericNode: xframe_core::GenericNode<Event = xframe_web::Event> {}
impl<T: xframe_core::GenericNode<Event = xframe_web::Event>> GenericNode for T {}

#[doc(inline)]
pub use xframe_reactive::*;

#[doc(inline)]
pub use xframe_web::{
    create_component, create_component as view, render, render_to_body, Component,
};

#[doc(inline)]
pub use xframe_macros::view;
