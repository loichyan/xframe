use crate::{utils::Visit, view_with};
use std::marker::PhantomData;
use xframe_core::{GenericComponent, GenericElement, GenericNode, IntoReactive, Reactive, View};
use xframe_reactive::Scope;

define_placeholder!(Placeholder("Placeholder for `xframe::Indexed` Component"));

pub struct Indexed<N, T, I, Children>
where
    N: GenericNode,
    I: 'static + Clone + Visit<T>,
    Children: Fn(&T) -> View<N>,
{
    cx: Scope,
    each: Option<Reactive<I>>,
    children: Option<Children>,
    marker: PhantomData<T>,
}

#[allow(non_snake_case)]
pub fn Indexed<N, T, I, Children>(cx: Scope) -> Indexed<N, T, I, Children>
where
    N: GenericNode,
    I: 'static + Clone + Visit<T>,
    Children: 'static + Fn(&T) -> View<N>,
{
    Indexed {
        cx,
        each: None,
        children: None,
        marker: PhantomData,
    }
}

impl<N, T, I, Children> Indexed<N, T, I, Children>
where
    N: GenericNode,
    I: 'static + Clone + Visit<T>,
    Children: 'static + Fn(&T) -> View<N>,
{
    pub fn build(self) -> impl GenericComponent<N> {
        let Self {
            cx, each, children, ..
        } = self;
        let each = each.expect("`each` was not provided");
        let children = children.expect("`each` was not provided");
        view_with(cx, move |placeholder: Placeholder<N>| {
            let placeholder = placeholder.into_node();
            let mut mounted = Vec::new();
            let dyn_view = cx.create_signal(View::Node(placeholder.clone()));
            cx.create_effect(move || {
                let each = each.clone().into_value();
                cx.untrack(|| {
                    let new_len = each.count();
                    let mounted_len = mounted.len();
                    if new_len == mounted_len {
                        return;
                    }
                    let current = dyn_view.get();
                    let mut last_node = current.last();
                    let parent = last_node.parent().unwrap_or_else(|| unreachable!());
                    if new_len == 0 {
                        if last_node.ne(&placeholder) {
                            // Replace with a placeholder.
                            let placeholder = View::Node(placeholder.clone());
                            current.replace_with(&parent, &placeholder);
                            mounted.clear();
                            dyn_view.set(placeholder);
                        }
                    } else if new_len > mounted_len {
                        // Append new views.
                        each.skip(mounted_len).visit(|val| {
                            let view = children(val);
                            view.move_after(&parent, &last_node);
                            if last_node.eq(&placeholder) {
                                parent.remove_child(&last_node);
                            }
                            last_node = view.last();
                            mounted.push(view);
                        });
                        dyn_view.set(View::from(mounted.clone()))
                    } else {
                        // Remove old views.
                        for view in mounted.drain(new_len..) {
                            view.remove_from(&parent);
                        }
                        dyn_view.set(View::from(mounted.clone()))
                    }
                });
            });
            View::from(dyn_view)
        })
    }

    pub fn each<E: IntoReactive<I>>(mut self, each: E) -> Self {
        if self.each.is_some() {
            panic!("`each` has already been provided");
        }
        self.each = Some(each.into_reactive());
        self
    }

    pub fn child(mut self, child: Children) -> Self {
        if self.children.is_some() {
            panic!("`child` has already been provided");
        }
        self.children = Some(child);
        self
    }
}
