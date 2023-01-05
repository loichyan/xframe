use xframe::{prelude::*, view, Else, GenericComponent, If, List, Scope, Switch, WebNode};

fn make_counter<N: WebNode>(cx: Scope, init: usize) -> impl GenericComponent<N> {
    let counter = cx.create_signal(init);
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
                .type_("button".s())
                .on_click(increment)
                "Click me: " (counter) " times!"
            }
        }
    }
}

fn main() {
    console_error_panic_hook::set_once();

    xframe::mount_to_body(|cx| {
        let counters = cx.create_signal(vec![1, 2, 3, 4]);
        let show = cx.create_signal(true);
        let push = move |_| counters.write(|x| x.push(x.len() + 1));
        let pop = move |_| {
            counters.write(|x| {
                x.pop();
            })
        };
        let clear = move |_| counters.write(Vec::clear);
        let toggle = move |_| show.update(|x| !*x);
        view! { cx,
            div {
                div {
                    button { .on_click(push) "Push" }
                    button { .on_click(pop) "Pop" }
                    button { .on_click(clear) "Clear" }
                    button { .on_click(toggle) "Toggle" }
                }
                If { .when(show) List { .each(counters) {|cx, &init| make_counter(cx, init)} } }
            }
        }
    });
}
