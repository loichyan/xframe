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

pub trait IntoEventHandler<Ev>: Sized {
    fn into_event_handler(self) -> EventHandler<Ev>;

    fn captured(self, val: bool) -> EventHandler<Ev> {
        let mut t = self.into_event_handler();
        t.options.capture = val;
        t
    }

    fn once(self, val: bool) -> EventHandler<Ev> {
        let mut t = self.into_event_handler();
        t.options.once = val;
        t
    }

    fn passive(self, val: bool) -> EventHandler<Ev> {
        let mut t = self.into_event_handler();
        t.options.passive = val;
        t
    }
}

impl<Ev, F> IntoEventHandler<Ev> for F
where
    F: 'static + FnMut(Ev),
{
    fn into_event_handler(self) -> EventHandler<Ev> {
        EventHandler {
            handler: Box::new(self),
            options: Default::default(),
        }
    }
}

impl<Ev> IntoEventHandler<Ev> for EventHandler<Ev> {
    fn into_event_handler(self) -> EventHandler<Ev> {
        self
    }
}
