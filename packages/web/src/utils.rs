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
                RenderOutput, View,
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

            #[allow(dead_code)]
            impl<N: GenericNode> $name<N> {
                fn new(cx: Scope) -> Self {
                    $name {
                        inner: Element::new(cx, || N::create($ty)),
                    }
                }

                fn render_with(
                    self,
                    f: impl 'static + FnMut(View<N>) -> Option<View<N>>,
                ) -> RenderOutput<N> {
                    self.inner.render_with(f)
                }
            }
        };
    };
}

macro_rules! define_placeholder {
    ($vis:vis struct $name:ident($desc:literal)) => {
        define_element!($vis struct $name(
            ::xframe_core::NodeType::Placeholder(std::borrow::Cow::Borrowed(
                if ::xframe_core::is_debug!() {
                    $desc
                } else {
                    "PLACEHOLDER"
                }
            ))
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
