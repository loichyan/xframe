use wasm_bindgen::JsCast;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Options {
    pub capture: bool,
    pub once: bool,
    pub passive: bool,
}

pub struct EventHandler<T> {
    pub handler: Box<dyn FnMut(T)>,
    pub options: Options,
}

impl<T> EventHandler<T> {
    pub fn cast<U>(mut self) -> EventHandler<U>
    where
        T: 'static + JsCast,
        U: 'static + JsCast,
    {
        EventHandler {
            handler: Box::new(move |ev: U| (self.handler)(ev.unchecked_into())),
            options: self.options,
        }
    }
}

pub trait IntoEventHandler<T>: Into<EventHandler<T>> {
    fn into_event_handler(self) -> EventHandler<T> {
        self.into()
    }

    fn captured(self, val: bool) -> EventHandler<T> {
        let mut t = self.into_event_handler();
        t.options.capture = val;
        t
    }

    fn once(self, val: bool) -> EventHandler<T> {
        let mut t = self.into_event_handler();
        t.options.once = val;
        t
    }

    fn passive(self, val: bool) -> EventHandler<T> {
        let mut t = self.into_event_handler();
        t.options.passive = val;
        t
    }
}

impl<T, U: Into<EventHandler<T>>> IntoEventHandler<T> for U {}

impl<T, F> From<F> for EventHandler<T>
where
    F: 'static + FnMut(T),
{
    fn from(t: F) -> Self {
        EventHandler {
            handler: Box::new(t),
            options: Default::default(),
        }
    }
}
