use wasm_bindgen::JsCast;
use xframe::{view, Fragment, Indexed, Keyed};

fn main() {
    console_error_panic_hook::set_once();

    // TODO: element ref
    xframe::render_to_body(|cx| {
        let todos = cx.create_signal(vec!["Do a diff?".to_owned()]);
        let current_input = cx.create_signal(String::new());
        let update_input = move |ev: xframe::Event| {
            // TODO: bind target type to associated element
            let el = ev
                .current_target()
                .unwrap()
                .unchecked_into::<xframe::element::HtmlInputElement>();
            current_input.set(el.value());
        };
        let add_todo = move |_| {
            let todo = current_input.get_untracked();
            todos.write(|t| {
                t.push(todo);
            });
        };
        let sort_todos = move |_| todos.write(|t| t.sort());
        let clear_todos = move |_| todos.write(|t| t.clear());
        view! { cx,
            Fragment {
                div {
                    label { .html_for("todo") "New todo:" }
                    input { .on_change(update_input) .id("todo") }
                }
                div {
                    button { .on_click(add_todo) "Add todo" }
                    button { .on_click(sort_todos) "Sort todos" }
                    button { .on_click(clear_todos) "Clear todos" }
                }
                ul {
                    Keyed {
                        .each(todos)
                        .key(Clone::clone)
                        { move |s| {
                            let s = s.clone();
                            view! { cx, li { (s) } }
                        } }
                    }
                }
            }
        }
    });
}
