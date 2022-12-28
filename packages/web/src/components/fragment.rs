use crate::element::GenericElement;
use xframe_core::{
    component::Fragment as FragmentBase, GenericComponent, GenericNode, RenderInput, RenderOutput,
};
use xframe_reactive::Scope;

define_placeholder!(struct Placeholder("PLACEHOLDER FOR `xframe::Fragment` COMPONENT"));

#[allow(non_snake_case)]
pub fn Fragment<N: GenericNode>(cx: Scope) -> Fragment<N> {
    GenericComponent::new(cx)
}

pub struct Fragment<N> {
    inner: FragmentBase<N>,
}

impl<N: GenericNode> GenericComponent<N> for Fragment<N> {
    fn new_with_input(input: RenderInput<N>) -> Self {
        Self {
            inner: FragmentBase::new_with_input(input),
        }
    }

    fn render_to_output(self) -> RenderOutput<N> {
        self.inner.render_to_output(Placeholder::<N>::TYPE)
    }
}

impl<N: GenericNode> Fragment<N> {
    pub fn child<C: GenericComponent<N>>(mut self, child: impl 'static + FnOnce(C) -> C) -> Self {
        self.inner.add_child(child);
        self
    }
}
