use crate::create_component;
use std::marker::PhantomData;
use wasm_bindgen::UnwrapThrowExt;
use xframe_core::{GenericComponent, GenericElement, GenericNode, IntoReactive, Reactive};
use xframe_reactive::Scope;

pub struct Show<N, When = Reactive<bool>, Fallback = Option<N>, Child = N> {
    cx: Scope,
    when: When,
    fallback: Fallback,
    child: Child,
    marker: PhantomData<N>,
}

impl<N> Show<N>
where
    N: GenericNode,
{
    pub fn render(self) -> impl GenericComponent<Node = N> {
        let Self {
            cx,
            when,
            fallback,
            child,
            ..
        } = self;
        create_component(cx, move |placeholder: crate::elements::placeholder<N>| {
            let mut current = placeholder.into_node();
            let parent = current
                .parent()
                .expect_throw("`Show` component must have a parent");
            cx.create_effect(move || {
                if when.clone().into_value() {
                    if current.ne(&child) {
                        parent.replace_child(&child, &current);
                        current = child.clone();
                    }
                } else if let Some(fallback) = fallback.as_ref() {
                    if current.ne(fallback) {
                        parent.replace_child(fallback, &current);
                        current = fallback.clone();
                    }
                }
            });
        })
    }
}

impl<N, When, Fallback, Child> Show<N, When, Fallback, Child> {
    pub fn when<T: IntoReactive<bool>>(self, when: T) -> Show<N, Reactive<bool>, Fallback, Child> {
        let Self {
            cx,
            fallback,
            child,
            marker,
            ..
        } = self;
        Show {
            cx,
            when: when.into_reactive(),
            fallback,
            child,
            marker,
        }
    }

    pub fn fallback<C: GenericComponent<Node = N>>(
        self,
        fallback: C,
    ) -> Show<N, When, Option<N>, Child> {
        let Self {
            cx,
            when,
            child,
            marker,
            ..
        } = self;
        Show {
            cx,
            when,
            fallback: Some(fallback.render()),
            child,
            marker,
        }
    }

    pub fn child<C: GenericComponent<Node = N>>(self, child: C) -> Show<N, When, Fallback, N> {
        let Self {
            cx,
            when,
            fallback,
            marker,
            ..
        } = self;
        Show {
            cx,
            when,
            fallback,
            child: child.render(),
            marker,
        }
    }
}

#[allow(non_snake_case)]
#[allow(clippy::type_complexity)]
pub fn Show<N: GenericNode>(cx: Scope) -> Show<N, (), Option<fn() -> N>, ()> {
    Show {
        cx,
        when: (),
        fallback: None,
        child: (),
        marker: PhantomData,
    }
}
