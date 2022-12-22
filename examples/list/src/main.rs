use xframe::{view, Else, If, List, Switch};

fn main() {
    console_error_panic_hook::set_once();

    xframe::render_to_body(|cx| {
        let counters = cx.create_signal(vec![1, 2, 3, 4]);
        let make_counter = move |&initial: &usize| {
            let counter = cx.create_signal(initial);
            let increment = move |_| counter.update(|x| *x + 1);
            let is_even = move || counter.get() % 2 == 0;
            view! { cx,
                fieldset {
                    div {
                        "Number " (counter) " is "
                        Switch {
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
        };
        let new_counter = move |_| counters.write(|x| x.push(x.len() + 1));
        let remove_counter = move |_| {
            counters.write(|x| {
                x.pop();
            })
        };
        view! { cx,
            div {
                div { button {
                    .on_click(new_counter)
                    "Click me to add new a counter"
                } }
                div { button {
                    .on_click(remove_counter)
                    "Click me to remove the last counter"
                } }
                List { .each(counters) {make_counter} }
            }
        }
    });
}
