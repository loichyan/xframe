use xframe_core::{Attribute, GenericNode, IntoReactive};
use xframe_reactive::Scope;

define_element!(
    pub struct text(NodeType::Text(std::borrow::Cow::Borrowed("")))
);

impl<N: GenericNode> text<N> {
    pub fn data<A: IntoReactive<Attribute>>(self, data: A) -> Self {
        self.inner.set_inner_text(data.into_reactive());
        self
    }
}

pub fn text<N: GenericNode>(cx: Scope) -> text<N> {
    text::new(cx)
}
