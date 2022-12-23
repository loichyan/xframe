use std::ops::RangeBounds;

use crate::{utils::Visit, Element};
use xframe_core::{
    view::ViewParentExt, GenericComponent, GenericElement, GenericNode, IntoReactive, Reactive,
    View,
};
use xframe_reactive::{untrack, Scope, ScopeDisposer};

define_placeholder!(Placeholder("PLACEHOLDER FOR `xframe::List` COMPONENT"));

pub struct List<N, T, I>
where
    I: 'static + Visit<T>,
{
    cx: Scope,
    each: Option<Reactive<I>>,
    children: Option<Box<dyn Fn(Scope, &T) -> View<N>>>,
}

#[allow(non_snake_case)]
pub fn List<N, T, I>(cx: Scope) -> List<N, T, I>
where
    N: GenericNode,
    I: 'static + Clone + Visit<T>,
{
    List {
        cx,
        each: None,
        children: None,
    }
}

impl<N, T, I> List<N, T, I>
where
    N: GenericNode,
    T: 'static,
    I: 'static + Clone + Visit<T>,
{
    pub fn build(self) -> impl GenericComponent<N> {
        let Self { cx, each, children } = self;
        let each = each.expect("`List::each` was not specified");
        let fn_view = children.expect("`List::child` was not specified");
        Element(cx).with_view(move |placeholder: Placeholder<N>| {
            let placeholder = View::node(placeholder.into_node());
            let dyn_view = View::dyn_(cx, placeholder.clone());
            cx.create_effect({
                let dyn_view = dyn_view.clone();
                let mut current_fragment = Fragment::<N>::default();
                move || {
                    // Only `each` needs to be tracked.
                    let each = each.clone().into_value();
                    untrack(|| {
                        let current_view = dyn_view.get();
                        let parent = current_view.parent();
                        let current_len = current_fragment.len();
                        let mut new_len = 0;
                        let next_sibling = current_view.next_sibling();
                        each.visit(|val| {
                            if new_len >= current_len {
                                // Append new views.
                                let (view, disposer) = cx.create_child(|cx| fn_view(cx, val));
                                parent.insert_before(&view, next_sibling.as_ref());
                                current_fragment.push(view, disposer);
                            }
                            new_len += 1;
                        });
                        let new_view;
                        if new_len == current_len {
                            return;
                        } else if new_len == 0 {
                            if current_len == 0 {
                                return;
                            }
                            // Replace empty view with a placeholder.
                            parent.replace_child(&placeholder, &current_view);
                            current_fragment.clear();
                            new_view = placeholder.clone();
                        } else {
                            if new_len < current_len {
                                // Remove extra views.
                                for (view, _) in current_fragment.drain(new_len..) {
                                    parent.remove_child(&view);
                                }
                            } else if current_len == 0 {
                                // new_len > current_len，remove the placeholder.
                                parent.remove_child(&placeholder);
                            }
                            new_view = View::fragment(current_fragment.views.clone());
                        }
                        debug_assert!(new_view.check_mount_order());
                        dyn_view.set(new_view);
                    });
                }
            });
            View::from(dyn_view)
        })
    }

    pub fn each<E: IntoReactive<I>>(mut self, each: E) -> Self {
        if self.each.is_some() {
            panic!("`List::each` has already been specified");
        }
        self.each = Some(each.into_reactive());
        self
    }

    pub fn child<C: GenericComponent<N>>(
        mut self,
        child: impl 'static + Fn(Scope, &T) -> C,
    ) -> Self {
        if self.children.is_some() {
            panic!("`List::child` has already been specified");
        }
        self.children = Some(Box::new(move |cx, val| child(cx, val).render()));
        self
    }
}

struct Fragment<N> {
    // It should be faster to clone the whole Vec.
    views: Vec<View<N>>,
    disposers: Vec<ScopeDisposer>,
}

impl<N> Default for Fragment<N> {
    fn default() -> Self {
        Self {
            views: Vec::new(),
            disposers: Vec::new(),
        }
    }
}

impl<N> Fragment<N> {
    fn len(&self) -> usize {
        self.views.len()
    }

    fn push(&mut self, view: View<N>, disposer: ScopeDisposer) {
        self.views.push(view);
        self.disposers.push(disposer);
    }

    fn drain<R: RangeBounds<usize> + Clone>(
        &mut self,
        range: R,
    ) -> impl '_ + Iterator<Item = (View<N>, ScopeDisposer)> {
        self.views
            .drain(range.clone())
            .zip(self.disposers.drain(range))
    }

    fn clear(&mut self) {
        self.views.clear();
        self.disposers.clear();
    }
}
