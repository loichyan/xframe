use std::marker::PhantomData;
use xframe_core::{Attribute, GenericElement, GenericNode, IntoAttribute};
use xframe_reactive::Scope;

pub struct Text<N, Data = Attribute> {
    cx: Scope,
    node: PhantomData<N>,
    data: Data,
}

impl<N> Text<N, ()> {
    pub fn data<A: IntoAttribute>(self, data: A) -> Text<N, Attribute> {
        Text {
            cx: self.cx,
            node: PhantomData,
            data: data.into_attribute(),
        }
    }
}

impl<N: GenericNode> GenericElement for Text<N> {
    type Node = N;
    fn into_node(self) -> Self::Node {
        let node = N::create_text_node("");
        self.cx.create_effect({
            let node = node.clone();
            move |_| node.set_inner_text(self.data.to_string().as_str())
        });
        node
    }
}

pub fn text<N: GenericNode>(cx: Scope) -> Text<N, ()> {
    Text {
        cx,
        node: PhantomData,
        data: (),
    }
}
