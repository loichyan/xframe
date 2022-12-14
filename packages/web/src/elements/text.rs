use xframe_core::{GenericNode, IntoReactive, RcStr, StringLike};
use xframe_reactive::Scope;

define_element!(
    pub struct text(NodeType::Text(RcStr::Literal("")))
);

impl<N: GenericNode> text<N> {
    pub fn data<A: IntoReactive<StringLike>>(self, data: A) -> Self {
        self.inner.set_inner_text(data.into_reactive());
        self
    }
}

pub fn text<N: GenericNode>(cx: Scope) -> text<N> {
    text::new(cx)
}
