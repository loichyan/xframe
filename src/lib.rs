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
    xframe_core::{component, prelude::*, template},
    xframe_macros::view,
    xframe_reactive::*,
    xframe_web::prelude::*,
};

/// A trait alias of [`xframe_core::GenericNode`].
pub trait GenericNode: xframe_core::GenericNode<Event = xframe_web::Event> {}
impl<T: xframe_core::GenericNode<Event = xframe_web::Event>> GenericNode for T {}

#[doc(hidden)]
#[path = "private.rs"]
pub mod __private;

/// Generate a unique [`TemplateId`] with the module path and line/column info.
///
/// [`TemplateId`]: template::TemplateId
#[macro_export]
macro_rules! id {
    () => {{
        thread_local! {
            static __ID: $crate::template::TemplateId =
                $crate::template::TemplateId::generate(concat!(module_path!(), ":", line!(), ":", column!()));
        }
        __ID.with(Clone::clone)
    }};
}
