use crate::{component::GlobalMode, GenericNode};
use std::cell::{Cell, RefCell};

thread_local! {
    static GLOBAL_ID: Cell<usize> = Cell::new(0);
}

pub struct GlobalState<N: 'static>(&'static std::thread::LocalKey<GlobalStateInner<N>>);

impl<N> GlobalState<N> {
    #[doc(hidden)]
    pub fn __new(val: &'static std::thread::LocalKey<GlobalStateInner<N>>) -> Self {
        Self(val)
    }
}

pub struct GlobalStateInner<N> {
    templates: RefCell<Vec<Option<Template<N>>>>,
    mode: RefCell<GlobalMode<N>>,
}

impl<N: GenericNode> Default for GlobalStateInner<N> {
    fn default() -> Self {
        Self {
            templates: Default::default(),
            mode: Default::default(),
        }
    }
}

impl<N: GenericNode> GlobalStateInner<N> {
    fn entry<U>(id: TemplateId, f: impl FnOnce(&mut Option<Template<N>>) -> U) -> U {
        let TemplateId { id, .. } = id;
        N::global_state().0.with(|global| {
            let templates = &mut global.templates.borrow_mut();
            if id >= templates.len() {
                templates.resize(id + 1, None);
            }
            f(&mut templates[id])
        })
    }

    pub(crate) fn get_template(id: TemplateId) -> Option<Template<N>> {
        Self::entry(id, |t| t.clone())
    }

    pub(crate) fn set_template(id: TemplateId, template: Template<N>) {
        Self::entry(id, |t| *t = Some(template));
    }

    pub(crate) fn mode<U>(f: impl FnOnce(&mut GlobalMode<N>) -> U) -> U {
        N::global_state()
            .0
            .with(|global| f(&mut *global.mode.borrow_mut()))
    }
}

#[derive(Clone, Copy)]
pub struct TemplateId {
    data: &'static str,
    id: usize,
}

impl TemplateId {
    pub fn generate(data: &'static str) -> Self {
        GLOBAL_ID.with(|global| {
            let id = global.get();
            global.set(id + 1);
            Self { id, data }
        })
    }

    pub fn data(&self) -> &'static str {
        self.data
    }
}

#[derive(Clone)]
pub(crate) struct Template<N> {
    pub container: N,
}
