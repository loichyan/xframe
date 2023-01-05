use rand::seq::SliceRandom;
use std::cell::Cell;
use wasm_bindgen::JsValue;
use web_sys::console;
use xframe::{prelude::*, view, For, If};

thread_local! {
    static COUNTER: Cell<usize> = Cell::new(0);
}

fn new_id() -> usize {
    COUNTER.with(|id| {
        let current = id.get();
        id.set(current + 1);
        current
    })
}

fn main() {
    console_error_panic_hook::set_once();

    let mut rng = rand::thread_rng();
    xframe::mount_to_body(|cx| {
        let ids = cx.create_signal(vec![]);
        cx.create_effect(move || {
            let arr = ids
                .get()
                .iter()
                .map(|&t| JsValue::from_f64(t as f64))
                .collect::<js_sys::Array>();
            console::log_1(&arr);
        });
        let show = cx.create_signal(true);
        let insert = move |_| {
            ids.write(|x| {
                let i = if x.is_empty() {
                    0
                } else {
                    rand::random::<usize>() % x.len()
                };
                x.insert(i, new_id());
            });
        };
        let remove = move |_| {
            ids.write(|x| {
                if !x.is_empty() {
                    let i = rand::random::<usize>() % x.len();
                    x.remove(i);
                }
            })
        };
        let shuffle = move |_| {
            ids.write(|x| {
                x.shuffle(&mut rng);
            })
        };
        let clear = move |_| ids.write(Vec::clear);
        let toggle = move |_| show.update(|x| !*x);
        view! { cx,
            div {
                div {
                    button { .on_click(insert) "Insert" }
                    button { .on_click(remove) "Remove" }
                    button { .on_click(shuffle) "Shuffle" }
                    button { .on_click(clear) "Clear" }
                    button { .on_click(toggle) "Toggle" }
                }
                If {
                    .when(show)
                    For {
                        .each(ids)
                        .key(|v| *v)
                        {move |cx, &id| view! { cx, [hr { } "ID: " (id.v())] }}
                    }
                }
            }
        }
    });
}
