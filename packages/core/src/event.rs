#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EventOptions {
    pub capture: bool,
}

pub struct EventHandler<Ev> {
    pub handler: Box<dyn FnMut(Ev)>,
    pub options: EventOptions,
}

impl<Ev: 'static> EventHandler<Ev> {
    pub fn cast<Ev2>(self) -> EventHandler<Ev2>
    where
        Ev2: 'static + Into<Ev>,
    {
        self.cast_with(Ev2::into)
    }

    pub fn cast_with<Ev2>(mut self, mut f: impl 'static + FnMut(Ev2) -> Ev) -> EventHandler<Ev2>
    where
        Ev2: 'static,
    {
        EventHandler {
            handler: Box::new(move |ev: Ev2| (self.handler)(f(ev))),
            options: self.options,
        }
    }
}

pub trait IntoEventHandler<Ev: 'static>: Sized {
    fn into_event_handler(self) -> EventHandler<Ev>;

    fn captured(self, val: bool) -> EventHandler<Ev> {
        let mut t = self.into_event_handler();
        t.options.capture = val;
        t
    }
}

impl<Ev: 'static> IntoEventHandler<Ev> for EventHandler<Ev> {
    fn into_event_handler(self) -> EventHandler<Ev> {
        self
    }
}

impl<Ev, F> IntoEventHandler<Ev> for F
where
    Ev: 'static,
    F: 'static + FnMut(Ev),
{
    fn into_event_handler(self) -> EventHandler<Ev> {
        EventHandler {
            handler: Box::new(self),
            options: Default::default(),
        }
    }
}
