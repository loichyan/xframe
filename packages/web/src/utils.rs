macro_rules! define_placeholder {
    ($vis:vis $name:ident($desc:literal)) => {
        #[derive(Clone)]
        $vis struct $name<N> {
            node: N,
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

pub trait UnwrapThrowValExt<T> {
    fn unwrap_throw_val(self) -> T;
}

impl<T> UnwrapThrowValExt<T> for Result<T, wasm_bindgen::JsValue> {
    fn unwrap_throw_val(self) -> T {
        self.unwrap_or_else(|e| wasm_bindgen::throw_val(e))
    }
}
