fn main() {
    use xframe_web::element::prelude::*;

    xframe_web::render_to_body(|cx| {
        let counter = cx.create_signal(0);
        div(cx).child(
            button(cx)
                .type_("button")
                .on_click(move |_| counter.update(|x| *x + 1))
                .child(text(cx).data("Click me: "))
                .child(text(cx).data(counter))
                .child(text(cx).data(" times!")),
        )
    })
}
