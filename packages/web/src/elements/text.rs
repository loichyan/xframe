use xframe_core::{Attribute, GenericNode, IntoReactive};
use xframe_reactive::Scope;

define_element!(
    pub struct text(NodeType::Text(std::borrow::Cow::Borrowed("")))
);

impl<N: GenericNode> text<N> {
    pub fn data<A: IntoReactive<Attribute>>(self, data: A) -> Self {
        let data = data.into_reactive(self.inner.cx);
        self.inner.cx.create_effect({
            let node = self.inner.root().clone();
            move || node.set_inner_text(&data.clone().into_value().into_string())
        });
        self
    }
}

pub fn text<N: GenericNode>(cx: Scope) -> text<N> {
    text::new(cx)
}
