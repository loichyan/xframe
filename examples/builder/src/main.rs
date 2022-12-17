use std::marker::PhantomData;
use xframe::element::prelude::*;
use xframe::{view, GenericComponent, GenericNode, Scope};

struct Counter<N> {
    cx: Scope,
    marker: PhantomData<N>,
}

impl<N: GenericNode> Counter<N> {
    pub fn build(self) -> impl GenericComponent<N> {
        let Self { cx, .. } = self;
        let counter = cx.create_signal(0);
        let increment = move |_| counter.update(|x| *x + 1);
        view(cx, move |div: div<_>| div).child(
            view(cx, move |e: button<_>| e.on_click(increment))
                .child_text("Click me: ")
                .child_text(counter)
                .child_text(" times!"),
        )
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

    xframe::render_to_body(|cx| {
        view! { cx,
             div {
                Counter {}
                Counter {}
            }
        }
    });
}
