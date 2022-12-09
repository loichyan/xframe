use web_sys::AddEventListenerOptions;

#[cfg(feature = "extra-events")]
#[doc(inline)]
pub use crate::generated::output::event_types::*;

pub struct EventHandlerWithOptions<Ev = web_sys::Event> {
    pub(crate) handler: Box<dyn FnMut(Ev)>,
    pub(crate) options: AddEventListenerOptions,
}

#[cfg(feature = "extra-events")]
const _: () = {
    use wasm_bindgen::JsCast;

    impl<Ev> EventHandlerWithOptions<Ev> {
        pub(crate) fn erase_type(self) -> EventHandlerWithOptions
        where
            Ev: 'static + wasm_bindgen::JsCast,
        {
            let Self {
                mut handler,
                options,
            } = self;
            EventHandlerWithOptions {
                handler: Box::new(move |ev: web_sys::Event| (handler)(ev.unchecked_into())),
                options,
            }
        }
    }
};

pub trait EventHandler<Ev>: Sized {
    fn into_event_handler(self) -> EventHandlerWithOptions<Ev>;

    fn captured(self, val: bool) -> EventHandlerWithOptions<Ev> {
        let mut t = self.into_event_handler();
        t.options.capture(val);
        t
    }

    fn once(self, val: bool) -> EventHandlerWithOptions<Ev> {
        let mut t = self.into_event_handler();
        t.options.once(val);
        t
    }

    fn passive(self, val: bool) -> EventHandlerWithOptions<Ev> {
        let mut t = self.into_event_handler();
        t.options.passive(val);
        t
    }
}

impl<Ev, F> EventHandler<Ev> for F
where
    F: 'static + FnMut(Ev),
{
    fn into_event_handler(self) -> EventHandlerWithOptions<Ev> {
        EventHandlerWithOptions {
            handler: Box::new(self),
            options: AddEventListenerOptions::default(),
        }
    }
}

impl<Ev> EventHandler<Ev> for EventHandlerWithOptions<Ev> {
    fn into_event_handler(self) -> EventHandlerWithOptions<Ev> {
        self
    }
}
