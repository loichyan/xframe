use std::marker::PhantomData;
use xframe::{view, Else, GenericComponent, GenericNode, If, Scope, Show};

struct Counter<N> {
    cx: Scope,
    marker: PhantomData<N>,
}

impl<N: GenericNode> Counter<N> {
    pub fn build(self) -> impl GenericComponent<Node = N> {
        let Self { cx, .. } = self;
        let counter = cx.create_signal(0);
        let increment = move |_| counter.update(|x| *x + 1);
        let is_even = move || counter.get() % 2 == 0;
        view! { cx,
            div {
                div {
                    "Number " (counter) " is "
                    Show {
                        If { .when(is_even) "even" }
                        Else { "odd" }
                    }
                    "."
                }
                button {
                    .type_("button")
                    .on_click(increment)
                    "Click me: " (counter) " times!"
                }
            }
        }
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
    xframe::render_to_body(|cx| {
        view! { cx,
             div {
                Counter {}
            }
        }
    })
}
