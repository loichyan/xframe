use xframe_core::{Attribute, GenericElement, GenericNode, IntoReactive};
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

impl<N: GenericNode> GenericElement for placeholder<N> {
    type Node = N;

    fn create(cx: Scope) -> Self {
        Self {
            cx,
            node: N::create_placeholder(""),
        }
    }

    fn create_with_node(cx: Scope, node: Self::Node) -> Self {
        Self { cx, node }
    }

    fn into_node(self) -> Self::Node {
        self.node
    }
}

pub fn placeholder<N: GenericNode>(cx: Scope) -> placeholder<N> {
    placeholder::create(cx)
}
