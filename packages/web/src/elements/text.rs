use std::borrow::Cow;
use xframe_core::{Attribute, GenericElement, GenericNode, IntoReactive, NodeType, View};
use xframe_reactive::Scope;

#[allow(non_camel_case_types)]
#[derive(Clone)]
pub struct text<N> {
    cx: Scope,
    node: N,
}

impl<N: GenericNode> From<text<N>> for View<N> {
    fn from(t: text<N>) -> Self {
        View::node(t.node)
    }
}

impl<N: GenericNode> text<N> {
    pub fn data<A: IntoReactive<Attribute>>(self, data: A) -> Self {
        let data = data.into_reactive(self.cx);
        self.cx.create_effect({
            let node = self.node.clone();
            move || node.set_inner_text(&data.clone().into_value().into_string())
        });
        text {
            cx: self.cx,
            node: self.node,
        }
    }
}

impl<N: GenericNode> GenericElement<N> for text<N> {
    const TYPE: NodeType = NodeType::Text(Cow::Borrowed(""));

    fn create_with_node(cx: Scope, node: N) -> Self {
        Self { cx, node }
    }

    fn into_node(self) -> N {
        self.node
    }
}

pub fn text<N: GenericNode>(cx: Scope) -> text<N> {
    text::create(cx)
}
