use crate::{GenericNode, View};
use ahash::AHashMap;
use std::{
    cell::{Cell, RefCell},
    fmt,
    rc::Rc,
};

thread_local! {
    static GLOBAL_ID: Cell<usize> = Cell::new(0);
}

pub struct Templates<N> {
    inner: Rc<RefCell<AHashMap<TemplateId, TemplateNode<N>>>>,
}

impl<N> Default for Templates<N> {
    fn default() -> Self {
        Self {
            inner: Default::default(),
        }
    }
}

impl<N> Clone for Templates<N> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<N: GenericNode> Templates<N> {
    pub(crate) fn get_or_insert_with(
        &self,
        id: TemplateId,
        f: impl FnOnce() -> TemplateNode<N>,
    ) -> TemplateNode<N> {
        let mut templates = self.inner.borrow_mut();
        let template = templates.entry(id).or_insert_with(f);
        TemplateNode {
            view: template.view.clone(),
            container: template.container.deep_clone(),
        }
    }
}

#[derive(Clone)]
pub(crate) struct TemplateNode<N> {
    pub view: View<N>,
    pub container: N,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TemplateId {
    id: usize,
}

impl Default for TemplateId {
    fn default() -> Self {
        Self::new()
    }
}

impl TemplateId {
    pub fn new() -> Self {
        Self {
            id: GLOBAL_ID.with(|id| {
                let current = id.get();
                id.set(current + 1);
                current
            }),
        }
    }
}

impl fmt::Display for TemplateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.id.fmt(f)
    }
}

pub struct Template<N> {
    pub init: TemplateInit<N>,
    pub render: TemplateRender<N>,
}

pub struct TemplateInit<N>(Box<dyn FnOnce() -> View<N>>);

impl<N> TemplateInit<N> {
    pub fn new(f: impl 'static + FnOnce() -> View<N>) -> Self {
        Self(Box::new(f))
    }

    pub fn init(self) -> View<N> {
        (self.0)()
    }
}

pub struct TemplateRender<N>(Box<dyn FnOnce(Option<N>) -> TemplateRenderOutput<N>>);

pub struct TemplateRenderOutput<N> {
    pub next_sibling: Option<N>,
    pub view: View<N>,
}

impl<N> TemplateRender<N> {
    pub fn new(f: impl 'static + FnOnce(Option<N>) -> TemplateRenderOutput<N>) -> Self {
        Self(Box::new(f))
    }

    pub fn render(self, node: Option<N>) -> TemplateRenderOutput<N> {
        (self.0)(node)
    }
}
