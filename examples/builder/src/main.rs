use std::marker::PhantomData;
use xframe::element::prelude::*;
use xframe::{view, GenericComponent, GenericNode, RenderOutput, Root, Scope, WebNode};

struct Counter<N> {
    cx: Scope,
    marker: PhantomData<N>,
}

impl<N: WebNode> GenericComponent<N> for Counter<N> {
    fn render(self) -> RenderOutput<N> {
        let Self { cx, .. } = self;
        let counter = cx.create_signal(0);
        let increment = move |_| counter.update(|x| *x + 1);
        Root(cx)
            .with(move || {
                div(cx).child(move || {
                    button(cx)
                        .on_click(increment)
                        .child_text("Click me: ")
                        .child_text(counter)
                        .child_text(" times!")
                })
            })
            .render()
    }
}

#[allow(non_snake_case)]
fn Counter<N: GenericNode>(cx: Scope) -> Counter<N> {
    Counter {
        cx,
        marker: PhantomData,
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
