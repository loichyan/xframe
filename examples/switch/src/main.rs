use xframe::{elements as xf, id, prelude::*, view, Else, If, Switch, TemplateId};

fn main() {
    console_error_panic_hook::set_once();

    xframe::mount_to_body(|cx| {
        let counter = cx.create_signal(1);
        let increment = move |_| counter.update(|x| *x + 1);
        let is_even = cx.create_memo(move || counter.get() % 2 == 0);
        let is_divisor_of = move |n: usize, id: fn() -> TemplateId| {
            let when = cx.create_memo(move || counter.get() % n == 0);
            move || {
                view! { cx,
                    If {
                        .id(id)
                        .when(when)
                        div { "Number " (counter) " is also the divisor of " (n.s()) "." }
                    }
                }
            }
        };
        cx.create_memo(move || counter.get() % 2 == 0);
        view! { cx,
            div {
                button {
                    .type_(xf::ButtonTypeValue::Button.s())
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
                {is_divisor_of(3, id!())}
                {is_divisor_of(5, id!())}
                {is_divisor_of(7, id!())}
            }
        }
    });
}
