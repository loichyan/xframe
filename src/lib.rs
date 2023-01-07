pub mod prelude {
    #[doc(inline)]
    pub use {xframe_core::IntoReactiveValue, xframe_web::GenericElement};
}

pub mod elements {
    #[doc(inline)]
    #[cfg(feature = "extra-elements")]
    pub use xframe_extra::{element_types::*, elements::*};

    #[doc(inline)]
    pub use xframe_web::{element_types::*, elements::*};

    #[cfg(feature = "extra-attributes")]
    #[doc(inline)]
    pub use xframe_extra::attr_types::*;
}

// TODO: merge into `elements`
pub mod event {
    #[cfg(feature = "extra-events")]
    #[doc(inline)]
    pub use xframe_extra::event_types::*;
}

#[doc(inline)]
pub use {
    xframe_core::{component, prelude::*},
    xframe_macros::view,
    xframe_reactive::*,
    xframe_web::prelude::*,
};

#[doc(hidden)]
#[path = "private.rs"]
pub mod __private;

/// Generate a unique [`TemplateId`] with the module path and line/column info.
#[macro_export]
macro_rules! id {
    () => {{
        $crate::id!(concat!(module_path!(), ":", line!(), ":", column!()))
    }};
    ($data:expr) => {{
        fn __id() -> $crate::TemplateId {
            thread_local! {
                static __ID: $crate::TemplateId = $crate::TemplateId::generate($data);
            }
            __ID.with(Clone::clone)
        }
        __id
    }};
}
