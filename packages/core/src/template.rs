use crate::{GenericNode, View};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

thread_local! {
    static GLOBAL_ID: Cell<usize> = Cell::new(0);
}

pub struct GlobalTemplates<N> {
    inner: Rc<RefCell<Vec<Option<TemplateContent<N>>>>>,
}

impl<N> Default for GlobalTemplates<N> {
    fn default() -> Self {
        Self {
            inner: Default::default(),
        }
    }
}

impl<N> Clone for GlobalTemplates<N> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<N: GenericNode> GlobalTemplates<N> {
    pub fn new() -> Self {
        Self::default()
    }

    fn entry<U>(id: TemplateId, f: impl FnOnce(&mut Option<TemplateContent<N>>) -> U) -> U {
        let TemplateId { id, .. } = id;
        let templates = N::global_templates().inner;
        let templates = &mut *templates.borrow_mut();
        if id >= templates.len() {
            templates.resize_with(id + 1, || None);
        }
        f(&mut templates[id])
    }

    pub(crate) fn get(id: TemplateId) -> Option<TemplateContent<N>> {
        Self::entry(id, |t| t.clone())
    }

    pub(crate) fn set(id: TemplateId, template: TemplateContent<N>) {
        Self::entry(id, |t| *t = Some(template));
    }
}

#[derive(Clone, Copy)]
pub struct TemplateId {
    data: &'static str,
    id: usize,
}

impl TemplateId {
    pub fn generate(data: &'static str) -> Self {
        let id = GLOBAL_ID.with(|global| {
            let id = global.get();
            global.set(id + 1);
            id
        });
        // let templates = N::global_templates();
        // let templates = &mut templates.inner.borrow_mut();
        // let id = templates.len();
        // templates.push(None);
        Self { id, data }
    }

    pub fn data(&self) -> &'static str {
        self.data
    }
}

#[derive(Clone)]
pub(crate) struct TemplateContent<N> {
    pub container: N,
    pub dehydrated: View<N>,
}
