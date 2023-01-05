use std::marker::PhantomData;
use xframe::{elements::*, id, prelude::*, GenericComponent, RenderOutput, Root, Scope, WebNode};

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
            .id(id!())
            .child(move || {
                div(cx).child(move || {
                    button(cx)
                        .on_click(increment)
                        .child(("Click me: ".s(), counter, " times!".s()))
                })
            })
            .render()
    }
}

#[allow(non_snake_case)]
fn Counter<N: WebNode>(cx: Scope) -> Counter<N> {
    Counter {
        cx,
        marker: PhantomData,
    }
}

fn main() {
    console_error_panic_hook::set_once();

    xframe::mount_to_body(|cx| {
        Root(cx)
            .id(id!())
            .child(move || div(cx).child((move || Counter(cx), move || Counter(cx))))
    });
}
