use crate::Element;
use ahash::AHashMap;
use std::{hash::Hash, rc::Rc};
use xframe_core::{
    is_debug, view::ViewParentExt, GenericComponent, GenericElement, GenericNode, IntoReactive,
    Reactive, View,
};
use xframe_reactive::{untrack, Scope, ScopeDisposer};

define_placeholder!(Placeholder("PLACEHOLDER FOR `xframe::For` COMPONENT"));

pub struct For<N, T, K> {
    cx: Scope,
    each: Option<Reactive<Vec<T>>>,
    key: Option<Box<dyn Fn(&T) -> K>>,
    children: Option<Box<dyn Fn(Scope, &T) -> View<N>>>,
}

#[allow(non_snake_case)]
pub fn For<N, T, K>(cx: Scope) -> For<N, T, K>
where
    N: GenericNode,
{
    For {
        cx,
        each: None,
        key: None,
        children: None,
    }
}

impl<N, T, K> For<N, T, K>
where
    N: GenericNode,
    T: 'static + Clone,
    K: 'static + Clone + Eq + Hash,
{
    pub fn build(self) -> impl GenericComponent<N> {
        let Self {
            cx,
            each,
            key,
            children,
        } = self;
        let each = each.expect("`For::each` was not specified");
        let fn_key = key.expect("`For::key` was not specified");
        let fn_view = children.expect("`For::child` was not specified");
        Element(cx).with_view(move |placeholder: Placeholder<N>| {
            let placeholder = View::node(placeholder.into_node());
            let dyn_view = View::dyn_(cx, placeholder.clone());
            cx.create_effect({
                let dyn_view = dyn_view.clone();
                let mut current_vals = Vec::<T>::new();
                let mut current_fragment: Rc<[View<N>]> = Rc::new([]);
                let mut current_disposers = Vec::<Option<ScopeDisposer>>::new();
                move || {
                    let new_vals = each.clone().into_value();
                    untrack(|| {
                        let current_view = dyn_view.get();
                        let parent = current_view.parent();
                        let new_view;
                        if new_vals.is_empty() {
                            if current_fragment.is_empty() {
                                return;
                            }
                            // Replace empty view with placeholder.
                            parent.replace_child(&placeholder, &current_view);
                            current_fragment = Rc::new([]);
                            current_disposers.clear();
                            new_view = placeholder.clone();
                        } else {
                            let mut new_fragment;
                            let mut new_disposers;
                            if current_fragment.is_empty() {
                                new_fragment = Vec::with_capacity(new_vals.len());
                                new_disposers = Vec::with_capacity(new_vals.len());
                                let position = placeholder.first();
                                for val in new_vals.iter() {
                                    let (view, disposer) = cx.create_child(|cx| fn_view(cx, val));
                                    parent.insert_before(&view, Some(&position));
                                    new_fragment.push(view);
                                    new_disposers.push(Some(disposer));
                                }
                                // Remove the placeholder.
                                parent.remove_child(&placeholder);
                            } else {
                                // Diff two fragments.
                                (new_fragment, new_disposers) = reconcile(
                                    parent.as_ref(),
                                    &current_vals,
                                    &current_fragment,
                                    &mut current_disposers,
                                    &new_vals,
                                    &fn_key,
                                    |val| cx.create_child(|cx| fn_view(cx, val)),
                                    View::fragment_shared(Rc::new([placeholder.clone()])),
                                );
                            }
                            current_fragment = new_fragment.into_boxed_slice().into();
                            current_disposers = new_disposers;
                            new_view = View::fragment_shared(current_fragment.clone());
                        }
                        debug_assert!(new_view.check_mount_order());
                        current_vals = new_vals;
                        dyn_view.set(new_view);
                    });
                }
            });
            View::from(dyn_view)
        })
    }

    pub fn each<E: IntoReactive<Vec<T>>>(mut self, each: E) -> Self {
        if self.each.is_some() {
            panic!("`For::each` has already been provided");
        }
        self.each = Some(each.into_reactive());
        self
    }

    pub fn key(mut self, key: impl 'static + Fn(&T) -> K) -> Self {
        if self.key.is_some() {
            panic!("`For::child` has already been provided");
        }
        self.key = Some(Box::new(key));
        self
    }

    pub fn child<C: GenericComponent<N>>(
        mut self,
        child: impl 'static + Fn(Scope, &T) -> C,
    ) -> Self {
        if self.children.is_some() {
            panic!("`For::child` has already been provided");
        }
        self.children = Some(Box::new(move |cx, val| child(cx, val).render()));
        self
    }
}

// Modified from: https://github.com/sycamore-rs/sycamore/
#[allow(clippy::too_many_arguments)]
fn reconcile<N, K, V>(
    parent: Option<&N>,
    current_vals: &[V],
    current_fragment: &Rc<[View<N>]>,
    current_disposers: &mut [Option<ScopeDisposer>],
    new_vals: &[V],
    fn_key: impl Fn(&V) -> K,
    fn_view: impl Fn(&V) -> (View<N>, ScopeDisposer),
    dummy_view: View<N>,
) -> (Vec<View<N>>, Vec<Option<ScopeDisposer>>)
where
    N: GenericNode,
    K: Eq + Hash,
{
    let a_len = current_vals.len();
    let b_len = new_vals.len();
    let after_last = current_fragment.last().unwrap().next_sibling();

    let mut new_fragment = Vec::new();
    new_fragment.resize_with(new_vals.len(), || dummy_view.clone());
    let mut new_disposers = Vec::new();
    new_disposers.resize_with(new_vals.len(), || None::<ScopeDisposer>);

    let mut a_start = 0;
    let mut b_start = 0;
    let mut a_end = a_len;
    let mut b_end = b_len;
    let mut a_map = None::<AHashMap<K, usize>>;
    let mut b_map = None::<AHashMap<K, usize>>;

    macro_rules! init_a_map {
        () => {
            a_map.get_or_insert_with(|| {
                current_vals[a_start..a_end]
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (fn_key(v), a_start + i))
                    .collect()
            })
        };
    }

    macro_rules! init_b_map {
        () => {
            b_map.get_or_insert_with(|| {
                new_vals[b_start..b_end]
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (fn_key(v), b_start + i))
                    .collect()
            })
        };
    }

    macro_rules! take_or_new_views {
        ($pos:expr, $start:expr, $end:expr) => {
            let a_map = &*init_a_map!();
            for (i, val) in new_vals[$start..$end].iter().enumerate() {
                // Ignore moved views.
                if let Some((view, disposer)) = a_map
                    .get(&fn_key(val))
                    .map(|&i| {
                        current_disposers[i]
                            .take()
                            .map(|t| (current_fragment[i].clone(), t))
                    })
                    .unwrap_or_else(|| Some(fn_view(val)))
                {
                    parent.insert_before(&view, $pos);
                    new_fragment[b_start + i] = view;
                    new_disposers[b_start + i] = Some(disposer);
                }
            }
        };
    }

    macro_rules! take_view {
        ($ia:expr, $ib:expr) => {
            new_fragment[$ib] = current_fragment[$ia].clone();
            new_disposers[$ib] = current_disposers[$ia].take();
        };
    }

    while a_start < a_end || b_start < b_end {
        if a_start == a_end {
            // Append new views.
            let position = if b_end < b_len {
                // a: 0 4
                // b: 0 1 2 3 4
                //      ^^^^^
                Some(current_fragment[a_end].first())
            } else {
                // All nodes in `a` should be removed, so we need to save the
                // next sibling before the loop.
                // a: 1 2 3 4
                // b: 5 6 7 8
                //    ^^^^^^^
                after_last
            };
            take_or_new_views!(position.as_ref(), b_start, b_end);
            b_start = b_end;
            break;
        } else if b_start == b_end {
            // Remove extra views.
            for (view, val) in current_fragment[a_start..a_end]
                .iter()
                .zip(current_vals[a_start..a_end].iter())
            {
                if b_map.as_ref().map(|m| m.contains_key(&fn_key(val))) != Some(true) {
                    parent.remove_child(view);
                }
            }
            a_start = a_end;
            break;
        }

        let a_start_k = fn_key(&current_vals[a_start]);
        let a_end_k = fn_key(&current_vals[a_end - 1]);
        let b_start_k = fn_key(&new_vals[b_start]);
        let b_end_k = fn_key(&new_vals[b_end - 1]);
        let a_start_v = &current_fragment[a_start];
        let a_end_v = &current_fragment[a_end - 1];

        if a_start_k == b_start_k {
            // Skip common preifx.
            take_view!(a_start, b_start);
            a_start += 1;
            b_start += 1;
        } else if a_end_k == b_end_k {
            // Skip common suffix.
            a_end -= 1;
            b_end -= 1;
            take_view!(a_end, b_end);
        } else if a_start_k == b_end_k && a_end_k == b_start_k {
            // Swap backwards.
            let start_next = a_start_v.next_sibling();
            let end_next = a_end_v.next_sibling();
            parent.insert_before(a_start_v, end_next.as_ref());
            parent.insert_before(a_end_v, start_next.as_ref());
            a_end -= 1;
            b_end -= 1;
            take_view!(a_start, b_end);
            take_view!(a_end, b_start);
            a_start += 1;
            b_start += 1;
        } else {
            let b_map = &*init_b_map!();
            if let Some(&index) = b_map.get(&a_start_k) {
                if index > b_start && index < b_end {
                    // Insert new views.
                    // a: 4
                    // b: 1 2 3 4
                    //    ^^^^^
                    let position = a_start_v.first();
                    take_or_new_views!(Some(&position), b_start, index);
                    b_start = index;
                } else {
                    // Ignore inserted views.
                    // a: 7 5
                    //      ^
                    // b: 5 6 7
                    a_start += 1;
                }
            } else {
                // Remove deleted views.
                parent.remove_child(a_start_v);
                a_start += 1;
            }
        }
    }

    if is_debug!() {
        for view in new_fragment.iter() {
            assert!(!dummy_view.ref_eq(view));
        }
        assert_eq!(new_fragment.len(), new_disposers.len());
        assert_eq!(a_start, a_end);
        assert_eq!(b_start, b_end);
    }
    (new_fragment, new_disposers)
}
