use crate::{utils::Visit, view_with};
use smallvec::SmallVec;
use xframe_core::{GenericComponent, GenericElement, GenericNode, IntoReactive, Reactive, View};
use xframe_reactive::{untrack, Scope};

const INITIAL_VIEW_SLOTS: usize = 4;

define_placeholder!(Placeholder("Placeholder for `xframe::Indexed` Component"));

pub struct Indexed<N, T, I>
where
    I: 'static + Visit<T>,
{
    cx: Scope,
    each: Option<Reactive<I>>,
    children: Option<Box<dyn Fn(&T) -> View<N>>>,
}

// TODO: rename to `List`
#[allow(non_snake_case)]
pub fn Indexed<N, T, I>(cx: Scope) -> Indexed<N, T, I>
where
    N: GenericNode,
    I: 'static + Clone + Visit<T>,
{
    Indexed {
        cx,
        each: None,
        children: None,
    }
}

impl<N, T, I> Indexed<N, T, I>
where
    N: GenericNode,
    T: 'static,
    I: 'static + Clone + Visit<T>,
{
    pub fn build(self) -> impl GenericComponent<N> {
        let Self { cx, each, children } = self;
        let each = each.expect("`each` was not provided");
        let fn_view = children.expect("`each` was not provided");
        view_with(cx, move |placeholder: Placeholder<N>| {
            let placeholder = placeholder.into_node();
            let mut mounted_views = SmallVec::<[_; INITIAL_VIEW_SLOTS]>::new();
            let dyn_view = cx.create_signal(View::Node(placeholder.clone()));
            cx.create_effect(move || {
                let each = each.clone().into_value();
                untrack(|| {
                    let current_view = dyn_view.get();
                    let current_last = current_view.last();
                    let parent = current_last.parent().unwrap();
                    let next_first = current_last.next_sibling();

                    let mounted_len = mounted_views.len();
                    let mut new_len = 0;
                    each.visit(|val| {
                        // Append new views.
                        if new_len >= mounted_len {
                            let view = fn_view(val);
                            view.move_before(&parent, next_first.as_ref());
                            mounted_views.push(view);
                        }
                        new_len += 1;
                    });
                    if new_len == mounted_len {
                        return;
                    }

                    if new_len == 0 {
                        if current_last.ne(&placeholder) {
                            // Replace with a placeholder.
                            let placeholder = View::Node(placeholder.clone());
                            current_view.replace_with(&parent, &placeholder);
                            mounted_views.clear();
                            dyn_view.set(placeholder);
                        }
                    } else {
                        if new_len < mounted_len {
                            // Remove old views.
                            for view in mounted_views.drain(new_len..) {
                                view.remove_from(&parent);
                            }
                        } else if current_last.eq(&placeholder) {
                            // Remove the placeholder.
                            parent.remove_child(&placeholder);
                        }
                        dyn_view.set(View::from(mounted_views.to_vec()))
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

    pub fn child<C: GenericComponent<N>>(mut self, child: impl 'static + Fn(&T) -> C) -> Self {
        if self.children.is_some() {
            panic!("`child` has already been provided");
        }
        self.children = Some(Box::new(move |t| child(t).render()));
        self
    }
}
