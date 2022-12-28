use crate::{GenericNode, View};
use std::{any::Any, cell::RefCell};

thread_local! {
    static TEMPLATES: GlobalTemplates = GlobalTemplates::default();
}

struct NoneTemplate;

#[derive(Default)]
pub(crate) struct GlobalTemplates {
    inner: RefCell<Vec<Box<dyn Any>>>,
}

impl GlobalTemplates {
    fn entry<U>(id: TemplateId, f: impl FnOnce(&mut Box<dyn Any>) -> U) -> U {
        let TemplateId { id, .. } = id;
        TEMPLATES.with(|templates| f(&mut templates.inner.borrow_mut()[id]))
    }

    pub(crate) fn get<N: GenericNode>(id: TemplateId) -> Option<TemplateContent<N>> {
        Self::entry(id, |t| {
            if t.downcast_ref::<NoneTemplate>().is_some() {
                None
            } else {
                Some(t.downcast_ref::<TemplateContent<N>>().unwrap().clone())
            }
        })
    }

    pub(crate) fn set<N: GenericNode>(id: TemplateId, template: TemplateContent<N>) {
        Self::entry(id, |t| *t = Box::new(template));
    }
}

#[derive(Clone, Copy)]
pub struct TemplateId {
    data: &'static str,
    id: usize,
}

impl TemplateId {
    pub fn generate(data: &'static str) -> Self {
        TEMPLATES.with(|templates| {
            let mut templates = templates.inner.borrow_mut();
            let id = templates.len();
            templates.push(Box::new(NoneTemplate));
            Self { id, data }
        })
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
