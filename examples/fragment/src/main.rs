use xframe::{view, Fragment, If, Show};

fn main() {
    xframe::render_to_body(|cx| {
        let counter = cx.create_signal(0);
        let increment = move |_| counter.update(|x| *x + 1);
        let is_even = move || counter.get() % 2 == 0;
        view! { cx,
            div {
                button {
                    .type_("button")
                    .on_click(increment)
                    "Click me!"
                }
                div { Show {
                    If {
                        .when(is_even)
                        Fragment {
                            "I'm only visible when " (counter) " is even."
                        }
                    }
                } }
            }
        }
    });
}
