use xframe::{view, Else, If, Switch};

fn main() {
    console_error_panic_hook::set_once();

    xframe::mount_to_body(|cx| {
        let counter = cx.create_signal(1);
        let increment = move |_| counter.update(|x| *x + 1);
        let is_even = cx.create_selector(move || counter.get() % 2 == 0);
        let is_divisor_of = move |n: usize| {
            let when = cx.create_selector(move || counter.get() % n == 0);
            move || {
                view! { cx,
                    If { .when(when) div { "Number " (counter) " is also the divisor of " (n) "." } }
                }
            }
        };
        cx.create_selector(move || counter.get() % 2 == 0);
        view! { cx,
            div {
                button {
                    .type_("button")
                    .on_click(increment)
                    "Click me: " (counter) " times!"
                }
                div {
                    "Number " (counter) " is "
                    Switch {
                        If { .when(is_even) "even" }
                        Else { "odd" }
                    }
                    "."
                }
                {is_divisor_of(3)}
                {is_divisor_of(5)}
                {is_divisor_of(7)}
            }
        }
    });
}
