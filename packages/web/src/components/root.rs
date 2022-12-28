use xframe_core::{
    component::Root as RootBase, GenericComponent, GenericNode, RenderInput, RenderOutput,
    TemplateId,
};
use xframe_reactive::Scope;

define_placeholder!(struct Placeholder("PLACEHOLDER FOR `xframe::Fragment` COMPONENT"));

#[allow(non_snake_case)]
pub fn Root<N: GenericNode>(cx: Scope) -> Root<N> {
    GenericComponent::new(cx)
}

pub struct Root<N> {
    input: RenderInput<N>,
    id: Option<fn() -> TemplateId>,
    children: Option<Box<dyn FnOnce(RenderInput<N>) -> RenderOutput<N>>>,
}

impl<N: GenericNode> GenericComponent<N> for Root<N> {
    fn new_with_input(input: RenderInput<N>) -> Self {
        Root {
            input,
            id: None,
            children: None,
        }
    }

    fn render_to_output(self) -> RenderOutput<N> {
        let Self {
            input,
            id,
            children,
        } = self;
        let children = children.expect("`Root::child` was not specified");
        let mut root = RootBase::new_with_input(input, children);
        if let Some(id) = id {
            root.set_id(id);
        }
        root.render_to_output()
    }
}

impl<N: GenericNode> Root<N> {
    pub fn input(mut self, input: RenderInput<N>) -> Self {
        self.input = input;
        self
    }

    pub fn input_from(mut self, other: Root<N>) -> Self {
        if other.id.is_some() || other.children.is_some() {
            panic!("`Root::input_from` only accept empty `Root`");
        }
        self.input = other.input;
        self
    }

    pub fn id(mut self, id: fn() -> TemplateId) -> Self {
        if self.id.is_some() {
            panic!("`Root::id` has been specified");
        }
        self.id = Some(id);
        self
    }

    pub fn child<C: GenericComponent<N>>(self, child: impl 'static + FnOnce(C) -> C) -> Self {
        self.child_impl(Box::new(|input| {
            child(C::new_with_input(input)).render_to_output()
        }))
    }

    fn child_impl(mut self, child: Box<dyn FnOnce(RenderInput<N>) -> RenderOutput<N>>) -> Self {
        if self.children.is_some() {
            panic!("`Root::child` has been specified");
        }
        self.children = Some(child);
        self
    }
}
