use crate::child::GenericChild;
use xframe_core::{
    component::Root as RootBase, is_debug, GenericComponent, GenericNode, NodeType, RenderOutput,
    TemplateId,
};
use xframe_reactive::Scope;

#[allow(non_snake_case)]
pub fn Root<N: GenericNode>(cx: Scope) -> Root<N> {
    Root {
        cx,
        id: None,
        children: None,
    }
}

pub struct Root<N> {
    cx: Scope,
    id: Option<fn() -> TemplateId>,
    children: Option<Box<dyn FnOnce() -> RenderOutput<N>>>,
}

impl<N: GenericNode> GenericComponent<N> for Root<N> {
    fn render(self) -> RenderOutput<N> {
        let Self { cx, id, children } = self;
        let children = children.expect("`Root::child` was not specified");
        let mut inner = RootBase::new(cx, children);
        if let Some(id) = id {
            inner.set_id(id);
        }
        inner.render(|data| {
            N::create(NodeType::Placeholder(if is_debug!() {
                format!("PLACEHOLDER FOR TEMPLATE<{data}>").into()
            } else {
                "PLACEHOLDER".into()
            }))
        })
    }
}

impl<N: GenericNode> Root<N> {
    pub fn id(mut self, id: fn() -> TemplateId) -> Self {
        if self.id.is_some() {
            panic!("`Root::id` has been specified");
        }
        self.id = Some(id);
        self
    }

    pub fn child(self, child: impl GenericChild<N>) -> Self {
        let cx = self.cx;
        self.child_impl(Box::new(move || child.render(cx)))
    }

    fn child_impl(mut self, child: Box<dyn FnOnce() -> RenderOutput<N>>) -> Self {
        if self.children.is_some() {
            panic!("`Root::child` has been specified");
        }
        self.children = Some(child);
        self
    }
}
