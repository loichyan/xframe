use xframe_core::{Attribute, GenericElement, GenericNode, IntoReactive, NodeType};
use xframe_reactive::Scope;

#[allow(non_camel_case_types)]
pub struct placeholder<N> {
    cx: Scope,
    node: N,
}

impl<N: GenericNode> placeholder<N> {
    pub fn desc<A: IntoReactive<Attribute>>(self, desc: A) -> Self {
        let desc = desc.into_reactive();
        self.cx.create_effect({
            let node = self.node.clone();
            move || node.set_inner_text(desc.clone().into_value().into_string_only().as_str())
        });
        placeholder {
            cx: self.cx,
            node: self.node,
        }
    }
}

impl<N: GenericNode> GenericElement<N> for placeholder<N> {
    const TYPE: NodeType = NodeType::Placeholder;

    fn create_with_node(cx: Scope, node: N) -> Self {
        Self { cx, node }
    }

    fn into_node(self) -> N {
        self.node
    }
}

pub fn placeholder<N: GenericNode>(cx: Scope) -> placeholder<N> {
    placeholder::create(cx)
}
