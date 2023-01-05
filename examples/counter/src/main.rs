use std::marker::PhantomData;
use xframe::{prelude::*, view, GenericComponent, IntoReactiveValue, RenderOutput, Scope, WebNode};

struct Counter<N> {
    cx: Scope,
    maker: PhantomData<N>,
}

impl<N: WebNode> GenericComponent<N> for Counter<N> {
    fn render(self) -> RenderOutput<N> {
        let Self { cx, .. } = self;

        let counter = cx.create_signal(0);
        let increment = move |_| counter.update(|x| *x + 1);

        view! { cx,
            div {
                button {
                    .type_("button".s())
                    .on_click(increment)
                    "Click me: " (counter) " times!"
                }
            }
        }
        .render()
    }
}

#[allow(non_snake_case)]
fn Counter<N: WebNode>(cx: Scope) -> Counter<N> {
    Counter {
        cx,
        maker: PhantomData,
    }
}

fn main() {
    console_error_panic_hook::set_once();

    xframe::mount_to_body(|cx| {
        view! { cx,
            div {
                Counter {}
                Counter {}
            }
        }
    });
}
