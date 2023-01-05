use xframe::{prelude::*, view, If, IntoReactiveValue, Switch};

fn main() {
    console_error_panic_hook::set_once();

    xframe::mount_to_body(|cx| {
        let counter = cx.create_signal(0);
        let increment = move |_| counter.update(|x| *x + 1);
        let is_even = move || counter.get() % 2 == 0;
        view! { cx,
            div {
                button {
                    .type_("button".s())
                    .on_click(increment)
                    "Click me!"
                }
                div { Switch {
                    If {
                        .when(is_even)
                        ["I'm only visible when " (counter) " is even."]
                    }
                } }
            }
        }
    });
}
