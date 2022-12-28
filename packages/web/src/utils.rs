macro_rules! define_element {
    ($(#[$attr:meta])* $vis:vis struct $name:ident($ty:expr)) => {
        $(#[$attr])*
        #[allow(non_camel_case_types)]
        $vis struct $name<N> {
            inner: ::xframe_core::component::Element<N>,
        }

        const _: () = {
            use crate::element::GenericElement;
            use ::xframe_core::{
                component::Element, GenericComponent, GenericNode, NodeType,
                RenderOutput,
            };
            use ::xframe_reactive::Scope;

            impl<N: GenericNode> AsRef<Element<N>> for $name<N> {
                fn as_ref(&self) -> &Element<N> {
                    &self.inner
                }
            }

            impl<N: GenericNode> AsMut<Element<N>> for $name<N> {
                fn as_mut(&mut self) -> &mut Element<N> {
                    &mut self.inner
                }
            }

            impl<N: GenericNode> GenericComponent<N> for $name<N> {
                fn render(self) -> RenderOutput<N> {
                    self.inner.render()
                }
            }

            impl<N: GenericNode> GenericElement<N> for $name<N> {
                const TYPE: NodeType = $ty;
            }

            impl<N: GenericNode> $name<N> {
                #[allow(dead_code)]
                $vis fn new(cx: Scope) -> Self {
                    $name {
                        inner: Element::new(cx, $ty),
                    }
                }
            }
        };
    };
}

macro_rules! define_placeholder {
    ($vis:vis struct $name:ident($desc:literal)) => {
        define_element!($vis struct $name(
            ::xframe_core::NodeType::Placeholder(std::borrow::Cow::Borrowed($desc))
        ));
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
