mod generated;
mod input;

#[cfg(feature = "elements")]
pub mod elements {
    #[doc(inline)]
    pub use crate::generated::output::elements::*;
}

#[cfg(feature = "elements")]
pub mod element_types {
    #[doc(inline)]
    pub use crate::generated::output::element_types::*;
}

#[cfg(feature = "attributes")]
pub mod attr_types {
    #[doc(inline)]
    pub use crate::generated::output::attr_types::*;
}

#[cfg(feature = "events")]
pub mod event_types {
    #[doc(inline)]
    pub use crate::generated::output::event_types::*;
}
