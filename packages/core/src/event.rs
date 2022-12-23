#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EventOptions {
    pub capture: bool,
    pub once: bool,
    pub passive: bool,
}

pub struct EventHandler<Ev> {
    pub handler: Box<dyn FnMut(Ev)>,
    pub options: EventOptions,
}

pub trait IntoEventHandler<Ev: 'static>: Into<EventHandler<Ev>> {
    fn into_event_handler(self) -> EventHandler<Ev> {
        self.into()
    }

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

    fn cast<Ev2>(self) -> EventHandler<Ev2>
    where
        Ev2: 'static + Into<Ev>,
    {
        self.cast_with(Ev2::into)
    }

    fn cast_with<Ev2>(self, mut f: impl 'static + FnMut(Ev2) -> Ev) -> EventHandler<Ev2>
    where
        Ev2: 'static,
    {
        let mut handler = self.into_event_handler();
        EventHandler {
            handler: Box::new(move |ev: Ev2| (handler.handler)(f(ev))),
            options: handler.options,
        }
    }
}

impl<Ev: 'static, U: Into<EventHandler<Ev>>> IntoEventHandler<Ev> for U {}

impl<Ev, F> From<F> for EventHandler<Ev>
where
    F: 'static + FnMut(Ev),
{
    fn from(t: F) -> Self {
        EventHandler {
            handler: Box::new(t),
            options: Default::default(),
        }
    }
}
