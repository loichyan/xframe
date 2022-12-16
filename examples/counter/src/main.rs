use std::marker::PhantomData;
use xframe::{view, GenericComponent, GenericNode, Scope};

struct Counter<N> {
    cx: Scope,
    marker: PhantomData<N>,
}

impl<N: GenericNode> GenericComponent<N> for Counter<N> {
    fn id() -> Option<xframe::TemplateId> {
        thread_local! {
            static ID: xframe::TemplateId = xframe::TemplateId::new();
        }
        Some(ID.with(Clone::clone))
    }

    fn build_template(self) -> xframe::Template<N> {
        let Self { cx, .. } = self;
        let counter = cx.create_signal(0);
        let increment = move |_| counter.update(|x| *x + 1);

        view! { cx,
            div {
                button {
                    .type_("button")
                    .on_click(increment)
                    "Click me: " (counter) " times!"
                }
            }
        }
        .build_template()
    }
}

impl<N: GenericNode> Counter<N> {
    pub fn build(self) -> impl GenericComponent<N> {
        self
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
                Counter {}
            }
        }
    });
}
