pub mod element {
    pub mod prelude {
        #[cfg(feature = "extra-elements")]
        #[doc(inline)]
        pub use xframe_extra::elements::*;
        #[doc(inline)]
        pub use xframe_web::elements::*;
    }
    #[doc(inline)]
    #[cfg(feature = "extra-elements")]
    pub use xframe_extra::element_types::*;
    #[doc(inline)]
    pub use {prelude::*, xframe_web::element_types::*};
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
pub use {
    xframe_core::{
        Attribute, EventOptions, GenericComponent, GenericElement, IntoEventHandler, IntoReactive,
        Reactive, Value,
    },
    xframe_macros::view,
    xframe_reactive::*,
    xframe_web::{components::*, render, render_to_body, view},
};

/// A trait alias of [`xframe_core::GenericNode`].
pub trait GenericNode: xframe_core::GenericNode<Event = xframe_web::Event> {}
impl<T: xframe_core::GenericNode<Event = xframe_web::Event>> GenericNode for T {}
