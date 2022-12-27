use xframe::{view, GenericComponent, RenderInput, RenderOutput, Scope, WebNode};

struct Counter<N> {
    input: RenderInput<N>,
}

impl<N: WebNode> GenericComponent<N> for Counter<N> {
    fn new_with_input(input: RenderInput<N>) -> Self {
        Self { input }
    }

    fn render_to_output(self) -> RenderOutput<N> {
        let Self { input } = self;

        let cx = input.cx;
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
        .input(input)
        .render_to_output()
    }
}

#[allow(non_snake_case)]
fn Counter<N: WebNode>(cx: Scope) -> Counter<N> {
    GenericComponent::new(cx)
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
