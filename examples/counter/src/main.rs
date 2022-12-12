fn main() {
    use xframe_web::{element::prelude::*, view};

    xframe_web::render_to_body(|cx| {
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
    })
}
