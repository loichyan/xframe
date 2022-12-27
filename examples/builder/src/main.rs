// use std::marker::PhantomData;
// use xframe::element::prelude::*;
// use xframe::{view, GenericComponent, GenericNode, Scope, WebNode, Root};

// struct Counter<N> {
//     cx: Scope,
//     root: Root<N>,
// }

// impl<N: WebNode> GenericComponent<N> for Counter<N> {
//     fn new_with_input(input: xframe::component::Input<N>) -> Self {
//         Self {
//             cx,
//             root: Root::new_with_input(input),
//         }
//     }
// }

// impl<N: WebNode> Counter<N> {
//     pub fn build(self) -> impl GenericComponent<N> {
//         let Self { cx, .. } = self;
//         let counter = cx.create_signal(0);
//         let increment = move |_| counter.update(|x| *x + 1);
//         view(cx, div).child(
//             view(cx, button)
//                 .then(move |e| e.on_click(increment))
//                 .child_text("Click me: ")
//                 .child_text(counter)
//                 .child_text(" times!"),
//         )
//     }
// }

// #[allow(non_snake_case)]
// fn Counter<N: GenericNode>(cx: Scope) -> Counter<N> {
//     Counter {
//         cx,
//         marker: PhantomData,
//     }
// }

fn main() {
    // TODO: better builder API
    console_error_panic_hook::set_once();

    // xframe::mount_to_body(|cx| {
    //     view! { cx,
    //          div {
    //             Counter {}
    //             Counter {}
    //         }
    //     }
    // });
}
