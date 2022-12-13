#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EventOptions {
    pub capture: bool,
    pub once: bool,
    pub passive: bool,
}

pub struct EventHandler<T: 'static> {
    pub handler: Box<dyn FnMut(T)>,
    pub options: EventOptions,
}

pub trait IntoEventHandler<T: 'static>: Into<EventHandler<T>> {
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

    fn cast<U>(self) -> EventHandler<U>
    where
        U: 'static + Into<T>,
    {
        self.cast_with(U::into)
    }

    fn cast_with<U>(self, f: fn(U) -> T) -> EventHandler<U>
    where
        U: 'static,
    {
        let mut handler = self.into_event_handler();
        EventHandler {
            handler: Box::new(move |ev: U| (handler.handler)(f(ev))),
            options: handler.options,
        }
    }
}

impl<T: 'static, U: Into<EventHandler<T>>> IntoEventHandler<T> for U {}

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
