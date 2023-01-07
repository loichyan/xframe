use rand::seq::SliceRandom;
use std::cell::Cell;
use tracing::info;
use xframe::{id, prelude::*, view, For, If};

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
    tracing_wasm::set_as_global_default();

    let mut rng = rand::thread_rng();
    xframe::mount_to_body(|cx| {
        let ids = cx.create_signal(vec![]);
        let realtime_ids = cx.create_signal(vec![]);
        cx.create_effect(move || info!("ids: {ids:?}"));
        let show = cx.create_signal(true);
        let insert = move |_| {
            realtime_ids.write(|x| {
                let i = if x.is_empty() {
                    0
                } else {
                    rand::random::<usize>() % x.len()
                };
                x.insert(i, new_id());
            });
        };
        let remove = move |_| {
            realtime_ids.write(|x| {
                if !x.is_empty() {
                    let i = rand::random::<usize>() % x.len();
                    x.remove(i);
                }
            })
        };
        let shuffle = move |_| {
            realtime_ids.write(|x| {
                x.shuffle(&mut rng);
            })
        };
        let clear = move |_| realtime_ids.write(Vec::clear);
        let sync = move |_| ids.set(realtime_ids.get_untracked());
        let toggle = move |_| show.update(|x| !*x);
        view! { cx,
            div {
                div {
                    button { .on_click(insert) "Insert" }
                    button { .on_click(remove) "Remove" }
                    button { .on_click(shuffle) "Shuffle" }
                    button { .on_click(clear) "Clear" }
                    button { .on_click(sync) "Sync" }
                    button { .on_click(toggle) "Toggle" }
                }
                If { .when(show) .id(id!()) [
                    div { For {
                        .each(realtime_ids)
                        .key(|v| *v)
                        {move |cx, &id| view! { cx, [(id.v()) ", "] }}
                    } }
                    For {
                        .each(ids)
                        .key(|v| *v)
                        {move |cx, &id| view! { cx, [hr { } "ID: " (id.v())] }}
                    }
                    hr { }
                    div { "End" }
                ] }
            }
        }
    });
}
