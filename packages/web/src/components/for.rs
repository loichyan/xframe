use crate::{utils::Visit, Element};
use ahash::AHashMap;
use std::hash::Hash;
use xframe_core::{
    view::ViewParentExt, GenericComponent, GenericElement, GenericNode, IntoReactive, Reactive,
    View,
};
use xframe_reactive::{untrack, Scope, ScopeDisposer};

define_placeholder!(Placeholder("PLACEHOLDER FOR `xframe::For` COMPONENT"));

pub struct For<N, T, K, I>
where
    N: GenericNode,
    I: 'static + Visit<T>,
{
    cx: Scope,
    each: Option<Reactive<I>>,
    key: Option<Box<dyn Fn(&T) -> K>>,
    children: Option<Box<dyn Fn(Scope, &T) -> View<N>>>,
}

#[allow(non_snake_case)]
pub fn For<N, T, K, I>(cx: Scope) -> For<N, T, K, I>
where
    N: GenericNode,
    I: 'static + Clone + Visit<T>,
{
    For {
        cx,
        each: None,
        key: None,
        children: None,
    }
}

impl<N, T, K, I> For<N, T, K, I>
where
    N: GenericNode,
    T: 'static,
    K: 'static + Clone + Eq + Hash,
    I: 'static + Clone + Visit<T>,
    N: GenericNode,
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
                let mut current_fragment = Fragment::default();
                let mut cached_views = AHashMap::<K, Cached<N>>::new();
                move || {
                    let each = each.clone().into_value();
                    untrack(|| {
                        let current_view = dyn_view.get();
                        let parent = current_view.parent();
                        // Reuse cached or create new views.
                        let mut new_fragment = Fragment::default();
                        each.visit(|val| {
                            let key = fn_key(val);
                            let Cached { view, moved, .. } =
                                cached_views.entry(key.clone()).or_insert_with(|| {
                                    let (view, disposer) = cx.create_child(|cx| fn_view(cx, val));
                                    Cached {
                                        view,
                                        moved: false,
                                        disposer,
                                    }
                                });
                            // Skip duplicated keys.
                            if !*moved {
                                *moved = true;
                                new_fragment.push(view.clone(), key);
                            }
                        });
                        let new_view;
                        if new_fragment.is_empty() {
                            if current_fragment.is_empty() {
                                return;
                            }
                            // Replace empty view with placeholder.
                            parent.replace_child(&placeholder, &current_view);
                            current_fragment.clear();
                            cached_views.clear();
                            new_view = placeholder.clone();
                        } else {
                            new_view = View::fragment(new_fragment.views.clone());
                            if current_fragment.is_empty() {
                                // Replace placeholder directly.
                                parent.replace_child(&new_view, &placeholder);
                            } else {
                                // Diff two fragments.
                                reconcile(
                                    &mut cached_views,
                                    parent.as_ref(),
                                    &current_fragment,
                                    &new_fragment,
                                );
                            }
                            current_fragment = new_fragment;
                        }
                        // Reset cache state.
                        for v in cached_views.values_mut() {
                            debug_assert!(v.moved);
                            v.moved = false;
                        }
                        debug_assert_eq!(cached_views.len(), current_fragment.len());
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

struct Cached<N: GenericNode> {
    view: View<N>,
    moved: bool,
    #[allow(dead_code)]
    disposer: ScopeDisposer,
}

struct Fragment<N: GenericNode, K> {
    views: Vec<View<N>>,
    keys: Vec<K>,
}

impl<N: GenericNode, K> Default for Fragment<N, K> {
    fn default() -> Self {
        Self {
            views: Vec::new(),
            keys: Vec::new(),
        }
    }
}

impl<N: GenericNode, K> Fragment<N, K> {
    fn len(&self) -> usize {
        self.views.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn push(&mut self, view: View<N>, key: K) {
        self.views.push(view);
        self.keys.push(key);
    }

    fn clear(&mut self) {
        self.views.clear();
        self.keys.clear();
    }

    fn iter(&self, start: usize, end: usize) -> impl '_ + Iterator<Item = (&View<N>, &K)> {
        self.views[start..end]
            .iter()
            .zip(self.keys[start..end].iter())
    }
}

// Modified from: https://github.com/sycamore-rs/sycamore/
fn reconcile<N, K>(
    cached_views: &mut AHashMap<K, Cached<N>>,
    parent: Option<&N>,
    a: &Fragment<N, K>,
    b: &Fragment<N, K>,
) where
    N: GenericNode,
    K: Clone + Eq + Hash,
{
    let a_len = a.len();
    let b_len = b.len();
    let a_next = a.views.last().unwrap().next_sibling();

    let mut a_start = 0;
    let mut b_start = 0;
    let mut a_end = a_len;
    let mut b_end = b_len;
    let mut b_map = None::<AHashMap<K, usize>>;

    while a_start < a_end || b_start < b_end {
        if a_start == a_end {
            // Append new views.
            let position = if b_end < b_len {
                if b_start == 0 {
                    // a: 4
                    // b: 1 2 3 4
                    //    ^^^^^
                    Some(b.views[b_end].first())
                } else {
                    // a: 1 4
                    // b: 1 2 3 4
                    //      ^^^
                    b.views[b_start - 1].next_sibling()
                }
            } else {
                // All nodes in `a` should be removed, so we need to save the
                // next sibling before the loop.
                // a: 1 2 3 4
                // b: 5 6 7 8
                //    ^^^^^^^
                a_next
            };
            for view in b.views[b_start..b_end].iter() {
                parent.insert_before(view, position.as_ref());
            }
            b_start = b_end;
            break;
        } else if b_start == b_end {
            // Remove extra views.
            for (view, key) in a.iter(a_start, a_end) {
                if b_map.as_ref().map(|m| m.contains_key(key)) != Some(true) {
                    parent.remove_child(view);
                    cached_views.remove(key);
                }
            }
            a_start = a_end;
            break;
        }

        let a_start_k = &a.keys[a_start];
        let a_end_k = &a.keys[a_end - 1];
        let b_start_k = &b.keys[b_start];
        let b_end_k = &b.keys[b_end - 1];
        let a_start_v = &a.views[a_start];
        let a_end_v = &a.views[a_end - 1];

        if a_start_k == b_start_k {
            // Skip common preifx.
            a_start += 1;
            b_start += 1;
        } else if a_end_k == b_end_k {
            // Skip common suffix.
            a_end -= 1;
            b_end -= 1;
        } else if a_start_k == b_end_k && a_end_k == b_start_k {
            // Swap backwards.
            let start_next = a_start_v.next_sibling();
            let end_next = a_end_v.next_sibling();
            parent.insert_before(a_start_v, end_next.as_ref());
            parent.insert_before(a_end_v, start_next.as_ref());
            a_start += 1;
            b_start += 1;
            a_end -= 1;
            b_end -= 1;
        } else {
            let map = &*b_map.get_or_insert_with(|| {
                b.keys[b_start..b_end]
                    .iter()
                    .enumerate()
                    .map(|(i, k)| (k.clone(), b_start + i))
                    .collect()
            });
            if let Some(&index) = map.get(a_start_k) {
                if index > b_start && index < b_end {
                    // a: 4
                    // b: 1 2 3 4
                    //    ^^^^^
                    let position = a_start_v.first();
                    for view in b.views[b_start..index].iter() {
                        parent.insert_before(view, Some(&position));
                    }
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
                cached_views.remove(a_start_k);
                a_start += 1;
            }
        }
    }

    debug_assert_eq!(a_start, a_end);
    debug_assert_eq!(b_start, b_end);
}
