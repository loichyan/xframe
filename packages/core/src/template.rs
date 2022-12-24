use crate::{GenericNode, View};
use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    rc::Rc,
};

thread_local! {
    static GLOBAL_ID: Cell<usize> = Cell::new(0);
}

pub struct Template<N> {
    pub init: TemplateInit<N>,
    pub render: TemplateRender<N>,
}

/// Initializes a [`Template`]. It should return the same node tree after each call
/// and no side effect should be performed during the invoking.
pub struct TemplateInit<N> {
    inner: Box<dyn FnOnce() -> View<N>>,
}

impl<N> TemplateInit<N> {
    pub fn new(f: impl 'static + FnOnce() -> View<N>) -> Self {
        Self { inner: Box::new(f) }
    }

    pub fn init(self) -> View<N> {
        (self.inner)()
    }
}

/// Renders a initialized [`Template`]. It accepts the first node in the template
/// and should return the next sibling of the last node in the template. And the
/// behavior defined by [`BeforeRendering`] should be applied to each node in the
/// template.
pub struct TemplateRender<N> {
    inner: Box<dyn FnOnce(BeforeRendering<N>, N) -> RenderOutput<N>>,
}

impl<N> TemplateRender<N> {
    pub fn new(f: impl 'static + FnOnce(BeforeRendering<N>, N) -> RenderOutput<N>) -> Self {
        Self { inner: Box::new(f) }
    }

    pub fn render(self, before_rendering: BeforeRendering<N>, node: N) -> RenderOutput<N> {
        (self.inner)(before_rendering, node)
    }
}

pub enum BeforeRendering<'a, N> {
    AppendTo(&'a N),
    RemoveFrom(&'a N),
    Nothing,
}

impl<N: GenericNode> BeforeRendering<'_, N> {
    pub fn apply_to(&self, node: &N) {
        use BeforeRendering::*;
        match self {
            AppendTo(parent) => parent.append_child(node),
            RemoveFrom(parent) => parent.remove_child(node),
            Nothing => {}
        }
    }
}

impl<N> Clone for BeforeRendering<'_, N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<N> Copy for BeforeRendering<'_, N> {}

pub struct RenderOutput<N> {
    pub next: Option<N>,
    pub view: View<N>,
}

pub struct GlobalTemplates<N> {
    inner: Rc<RefCell<HashMap<usize, TemplateContent<N>>>>,
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

    pub(crate) fn clone_or_insert_with(
        id: TemplateId,
        f: impl FnOnce() -> TemplateContent<N>,
    ) -> TemplateContent<N> {
        N::global_templates()
            .inner
            .borrow_mut()
            .entry(id.id)
            .or_insert_with(f)
            .deep_clone()
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct TemplateId {
    data: &'static str,
    id: usize,
}

impl TemplateId {
    pub fn generate(data: &'static str) -> Self {
        Self {
            id: GLOBAL_ID.with(|global_id| {
                let id = global_id.get();
                global_id.set(id + 1);
                id
            }),
            data,
        }
    }

    pub fn data(&self) -> &'static str {
        self.data
    }
}

pub(crate) struct TemplateContent<N> {
    pub container: N,
}

impl<N: GenericNode> TemplateContent<N> {
    pub fn deep_clone(&self) -> Self {
        Self {
            container: self.container.deep_clone(),
        }
    }
}
