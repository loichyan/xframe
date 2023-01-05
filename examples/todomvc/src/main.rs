use wasm_bindgen::{JsCast, UnwrapThrowExt};
use xframe::{
    elements as el, event as ev, prelude::*, view, DomNode, GenericComponent, If, List, Scope,
    ScopeExt, Signal, WebNode,
};

#[derive(Clone)]
struct Todo {
    content: Signal<String>,
    editing: Signal<bool>,
    completed: Signal<bool>,
    removed: Signal<bool>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ShowMode {
    All,
    Active,
    Completed,
}

fn make_todo<N: WebNode>(
    cx: Scope,
    show_mode: Signal<ShowMode>,
    todo: &Todo,
) -> impl GenericComponent<N> {
    let Todo {
        content,
        editing,
        completed,
        removed,
    } = todo.clone();
    let toggle = move |_| completed.update(|x| !*x);
    let remove = move |_| removed.set(true);
    let show = move || {
        if removed.get() {
            false
        } else {
            match show_mode.get() {
                ShowMode::All => true,
                ShowMode::Active => !completed.get(),
                ShowMode::Completed => completed.get(),
            }
        }
    };
    let edit_input = cx.create_node_ref::<N>();
    let set_editing = move |_| {
        editing.set(true);
        if let Some(input) = edit_input.get_as::<DomNode>() {
            input
                .into_js()
                .unchecked_into::<el::Input>()
                .focus()
                .unwrap_throw();
        }
    };
    let save_editing = move |ev: ev::Blur| {
        if editing.get() {
            let input = ev.current_target().unwrap().unchecked_into::<el::Input>();
            content.set(input.value().trim().to_owned());
            editing.set(false);
        }
    };
    let done_editing = move |ev: ev::Keydown| {
        if ev.key() == "Enter" {
            save_editing(ev.unchecked_into());
        }
    };
    view! { cx,
        li {
            .class("todo")
            .classx("completed", completed)
            .classx("editing", editing)
            If {
                .when(show)
                [div {
                    .class("view")
                    input {
                        .class("toggle")
                        .type_("checkbox".s())
                        .checked(completed)
                        .on_input(toggle)
                    }
                    label { .on_dblclick(set_editing) (content) }
                    button { .class("destroy") .on_click(remove) }
                }
                If {
                    .when(editing)
                    input {
                        .class("edit")
                        .ref_(edit_input)
                        .value(content)
                        .on_focusout(save_editing)
                        .on_keydown(done_editing)
                    }
                }]
            }
        }
    }
}

fn main() {
    console_error_panic_hook::set_once();

    xframe::mount_to_body(|cx| {
        let todos = cx.create_signal(vec![]);
        let show_mode = cx.create_signal(ShowMode::All);

        let add_todo = move |ev: ev::Keydown| {
            if ev.key() == "Enter" {
                let input = ev.current_target().unwrap().unchecked_into::<el::Input>();
                let todo = Todo {
                    content: cx.create_signal(input.value().trim().to_owned()),
                    editing: cx.create_signal(false),
                    completed: cx.create_signal(false),
                    removed: cx.create_signal(false),
                };
                todos.write(|todos| todos.push(todo));
            }
        };
        let make_todo = move |cx, todo: &Todo| make_todo(cx, show_mode, todo);
        let remaining_count = cx.create_memo(move || {
            todos
                .get()
                .iter()
                .filter(|todo| !todo.removed.get())
                .count()
        });
        let filter_selected = move |mode: ShowMode| move || show_mode.get() == mode;
        let filter_set = move |mode: ShowMode| move |_| show_mode.set(mode);
        let clear_complted = move |_| {
            todos.get().iter().for_each(|todo| {
                if todo.completed.get() {
                    todo.removed.set(true);
                }
            })
        };

        view! { cx,
            section {
                .class("todoapp")
                header {
                    .class("header")
                    h1 { "Todos" }
                    input {
                        .class("new-todo")
                        .placeholder("What needs to be done?".s())
                        .on_keydown(add_todo)
                    }
                }
                If {
                    .when(move || !todos.get().is_empty())
                    section {
                        .class("main")
                        ul {
                            .class("todo-list")
                            List { .each(todos) {make_todo} }
                        }
                    }
                }
                footer {
                    .class("footer")
                    span {
                        .class("todo-count")
                        strong { (remaining_count) } " item"
                        (cx.create_memo(move || if remaining_count.get() > 1 { "s" } else { "" }))
                        " left"
                    }
                    ul {
                        .class("filters")
                        li { a {
                            .href("#/".s()) "All"
                            .classx("selected", filter_selected(ShowMode::All))
                            .on_click(filter_set(ShowMode::All))
                        } }
                        li { a {
                            .href("#/active".s()) "Active"
                            .classx("selected", filter_selected(ShowMode::Active))
                            .on_click(filter_set(ShowMode::Active))
                        } }
                        li { a {
                            .href("#/completed".s()) "Completed"
                            .classx("selected", filter_selected(ShowMode::Completed))
                            .on_click(filter_set(ShowMode::Completed))
                        } }
                    }
                    If {
                        .when(move || remaining_count.get() > 0)
                        button {
                            .class("clear-completed")
                            .on_click(clear_complted)
                            "Clear completed"
                        }
                    }
                }
            }
        }
    });
}
