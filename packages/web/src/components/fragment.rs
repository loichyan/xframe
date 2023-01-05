use crate::{child::GenericChild, element::GenericElement};
use xframe_core::{
    component::Fragment as FragmentBase, GenericComponent, GenericNode, RenderOutput,
};
use xframe_reactive::Scope;

define_placeholder!(struct Placeholder("PLACEHOLDER FOR `xframe::Fragment` COMPONENT"));

#[allow(non_snake_case)]
pub fn Fragment<N: GenericNode>(cx: Scope) -> Fragment<N> {
    Fragment::with_capacity(cx, 0)
}

pub struct Fragment<N> {
    inner: FragmentBase<N>,
}

impl<N: GenericNode> GenericComponent<N> for Fragment<N> {
    fn render(self) -> RenderOutput<N> {
        self.inner.render(|| N::create(Placeholder::<N>::TYPE))
    }
}

impl<N: GenericNode> Fragment<N> {
    pub fn with_capacity(cx: Scope, capacity: usize) -> Self {
        Self {
            inner: FragmentBase::with_capacity(cx, capacity),
        }
    }

    pub fn child(mut self, child: impl GenericChild<N>) -> Self {
        let cx = self.inner.cx;
        self.inner.add_child(move || child.render(cx));
        self
    }
}
