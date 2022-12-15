use xframe::{view, Else, If, Show};

fn main() {
    xframe::render_to_body(|cx| {
        let counter = cx.create_signal(0);
        let increment = move |_| counter.update(|x| *x + 1);
        let is_even = move || counter.get() % 2 == 0;
        view! { cx,
            div {
                div {
                    // FIXME: empty page if there are nodes before `Show`
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
    });
}
