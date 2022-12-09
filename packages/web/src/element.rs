use wasm_bindgen::{intern, JsCast};

use crate::GenericNode;

#[cfg(feature = "extra-elements")]
#[doc(inline)]
pub use crate::generated::output::{element_types::*, elements::*};

#[cfg(feature = "extra-elements")]
pub mod prelude {
    #[doc(inline)]
    pub use crate::generated::output::elements::*;
}

pub trait GenericElement: Sized {
    type Node: GenericNode;
    fn into_node(self) -> Self::Node;
}

pub(crate) struct BaseElement<N>(N);

impl<N: GenericNode> GenericElement for BaseElement<N> {
    type Node = N;
    fn into_node(self) -> Self::Node {
        self.0
    }
}

#[allow(dead_code)]
impl<N: GenericNode> BaseElement<N> {
    pub fn create(tag: &'static str) -> Self {
        Self(N::create(intern(tag).into()))
    }

    pub fn node(&self) -> &N {
        &self.0
    }

    pub fn as_web_sys_element<T>(&self) -> &T
    where
        N: AsRef<web_sys::Node>,
        T: JsCast,
    {
        self.0.as_ref().unchecked_ref()
    }

    pub fn set_attribute<T>(&self, name: &'static str, val: T)
    where
        T: crate::attr::IntoAttribute,
    {
        self.0
            .set_attribute(intern(name).into(), val.into_attribute());
    }

    pub fn listen_event<Ev>(
        &self,
        event: &'static str,
        handler: impl crate::event::EventHandler<Ev>,
    ) where
        Ev: 'static + JsCast,
    {
        self.0.listen_event(
            intern(event).into(),
            handler.into_event_handler().erase_type(),
        );
    }
}
