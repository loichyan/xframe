use crate::element::GenericElement;
use xframe_core::{
    component::Fragment as FragmentBase, GenericComponent, GenericNode, RenderOutput,
};
use xframe_reactive::Scope;

define_placeholder!(struct Placeholder("PLACEHOLDER FOR `xframe::Fragment` COMPONENT"));

#[allow(non_snake_case)]
pub fn Fragment<N: GenericNode>(cx: Scope) -> Fragment<N> {
    Fragment {
        inner: FragmentBase::new(cx),
    }
}

pub struct Fragment<N> {
    inner: FragmentBase<N>,
}

impl<N: GenericNode> GenericComponent<N> for Fragment<N> {
    fn render(self) -> RenderOutput<N> {
        self.inner.render(Placeholder::<N>::TYPE)
    }
}

impl<N: GenericNode> Fragment<N> {
    pub fn child<C: GenericComponent<N>>(
        mut self,
        child: impl 'static + FnOnce(Scope) -> C,
    ) -> Self {
        self.inner.add_child(child);
        self
    }
}
