macro_rules! define_placeholder {
    ($vis:vis $name:ident($desc:literal)) => {
        $vis struct $name<N> {
            node: N,
        }

        #[allow(dead_code)]
        impl<N: ::xframe_core::GenericNode> $name<N> {
            pub fn new() -> Self {
                Self {
                    node: N::create(<Self as ::xframe_core::GenericElement<N>>::TYPE),
                }
            }
        }

        impl<N: ::xframe_core::GenericNode> xframe_core::GenericElement<N> for $name<N> {
            const TYPE: ::xframe_core::NodeType =
                ::xframe_core::NodeType::Placeholder(::std::borrow::Cow::Borrowed($desc));

            fn create_with_node(_: ::xframe_reactive::Scope, node: N) -> Self {
                Self { node }
            }

            fn into_node(self) -> N {
                self.node
            }
        }
    };
}
