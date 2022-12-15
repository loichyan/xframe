use xframe_core::{Attribute, GenericElement, GenericNode, IntoReactive, NodeType};
use xframe_reactive::Scope;

#[allow(non_camel_case_types)]
pub struct text<N> {
    cx: Scope,
    node: N,
}

impl<N: GenericNode> text<N> {
    pub fn data<A: IntoReactive<Attribute>>(self, data: A) -> Self {
        let data = data.into_reactive();
        self.cx.create_effect({
            let node = self.node.clone();
            move || node.set_inner_text(data.clone().into_value().into_string_only().as_str())
        });
        text {
            cx: self.cx,
            node: self.node,
        }
    }
}

impl<N: GenericNode> GenericElement<N> for text<N> {
    const TYPE: NodeType = NodeType::Text;

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
