use std::{hash::Hash, rc::Rc};
use tracing::debug;
use xframe_core::{
    is_debug, view::ViewParentExt, GenericComponent, GenericNode, HashMap, IntoReactive, Reactive,
    RenderOutput, View,
};
use xframe_reactive::{untrack, Scope, ScopeDisposer};

define_placeholder!(struct Placeholder("PLACEHOLDER FOR `xframe::For` COMPONENT"));

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
    T: 'static + Clone,
    K: 'static + Clone + Eq + Hash,
{
    For {
        cx,
        each: None,
        key: None,
        children: None,
    }
}

impl<N, T, K> GenericComponent<N> for For<N, T, K>
where
    N: GenericNode,
    T: 'static + Clone,
    K: 'static + Clone + Eq + Hash,
{
    fn render(self) -> RenderOutput<N> {
        let Self {
            cx,
            each,
            key,
            children,
        } = self;
        let each = each.expect("`For::each` was not specified");
        let fn_key = key.expect("`For::key` was not specified");
        let fn_view = children.expect("`For::child` was not specified");
        let mut current_vals = Vec::<T>::new();
        let mut current_fragment: Rc<[View<N>]> = Rc::new([]);
        let mut current_disposers = Vec::<Option<ScopeDisposer>>::new();
        let mut placeholder = None;
        Placeholder::<N>::new(cx).render_with(move |current_view| {
            let placeholder = &*placeholder.get_or_insert_with(|| current_view.clone());
            let new_vals = each.clone().into_value();
            untrack(|| {
                let parent = current_view.parent();
                let new_view;
                if new_vals.is_empty() {
                    if current_fragment.is_empty() {
                        return None;
                    }
                    // Replace empty view with placeholder.
                    parent.replace_child(placeholder, &current_view);
                    current_fragment = Rc::new([]);
                    current_disposers = Vec::new();
                    new_view = placeholder.clone();
                } else {
                    let mut new_fragment;
                    let mut new_disposers;
                    if current_fragment.is_empty() {
                        new_fragment = Vec::with_capacity(new_vals.len());
                        new_disposers = Vec::with_capacity(new_vals.len());
                        // Append new views.
                        for val in new_vals.iter() {
                            let (view, disposer) = cx.create_child(|cx| fn_view(cx, val));
                            parent.insert_before(&view, Some(placeholder));
                            new_fragment.push(view);
                            new_disposers.push(Some(disposer));
                        }
                        // Remove the placeholder.
                        parent.remove_child(placeholder);
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
                Some(new_view)
            })
        })
    }
}

impl<N, T, K> For<N, T, K>
where
    N: GenericNode,
    T: 'static + Clone,
    K: 'static + Clone + Eq + Hash,
{
    pub fn each<E: IntoReactive<Vec<T>>>(mut self, each: E) -> Self {
        if self.each.is_some() {
            panic!("`For::each` has been provided");
        }
        self.each = Some(each.into_reactive());
        self
    }

    pub fn key(mut self, key: impl 'static + Fn(&T) -> K) -> Self {
        if self.key.is_some() {
            panic!("`For::child` has been provided");
        }
        self.key = Some(Box::new(key));
        self
    }

    pub fn child<C: GenericComponent<N>>(
        mut self,
        child: impl 'static + Fn(Scope, &T) -> C,
    ) -> Self {
        if self.children.is_some() {
            panic!("`For::child` has been provided");
        }
        self.children = Some(Box::new(move |cx, val| child(cx, val).render_view()));
        self
    }
}

// Modified from: https://github.com/localvoid/ivi/blob/a769a11/packages/ivi/src/vdom/reconciler.ts#L823
#[allow(clippy::too_many_arguments)]
fn reconcile<N, K, V>(
    parent: Option<&N>,
    // Old values.
    a_vals: &[V],
    a_views: &[View<N>],
    a_disposers: &mut [Option<ScopeDisposer>],
    // New values.
    b_vals: &[V],
    fn_key: impl Fn(&V) -> K,
    fn_view: impl Fn(&V) -> (View<N>, ScopeDisposer),
    dummy_view: View<N>,
) -> (Vec<View<N>>, Vec<Option<ScopeDisposer>>)
where
    N: GenericNode,
    K: Eq + Hash,
{
    if is_debug!() {
        assert_eq!(a_vals.len(), a_views.len());
        assert_eq!(a_vals.len(), a_disposers.len());
    }

    let mut a_end = a_vals.len();
    let mut b_end = b_vals.len();
    let mut start = 0;
    let after_last = a_views.last().unwrap().next_sibling().map(View::node);

    let mut b_views = Vec::new();
    b_views.resize_with(b_vals.len(), || dummy_view.clone());
    let mut b_disposers = Vec::new();
    b_disposers.resize_with(b_vals.len(), || None::<ScopeDisposer>);

    macro_rules! take_a {
        ($ia:expr, $ib:expr) => {{
            let (ia, ib) = ($ia, $ib);
            b_views[ib] = a_views[ia].clone();
            ::std::mem::swap(&mut a_disposers[ia], &mut b_disposers[ib]);
        }};
    }

    macro_rules! remove_a {
        ($i:expr) => {{
            let i = $i;
            parent.remove_child(&a_views[i]);
            drop(a_disposers[i].take());
        }};
    }

    macro_rules! get_next_b {
        ($i:expr) => {{
            let i = $i;
            b_views.get(i + 1).or_else(|| after_last.as_ref())
        }};
    }

    macro_rules! insert_b {
        ($i:expr) => {{
            let i = $i;
            insert_b!(i, get_next_b!(i))
        }};
        ($i:expr, $next:expr) => {{
            let i = $i;
            let next = $next;
            parent.insert_before(&b_views[i], next);
        }};
    }

    macro_rules! new_b {
        ($i:expr) => {{
            let i = $i;
            let (view, disposer) = fn_view(&b_vals[i]);
            b_views[i] = view;
            b_disposers[i] = Some(disposer);
            insert_b!(i);
        }};
    }

    // Skip common prefix.
    for (a, b) in a_vals[start..].iter().zip(b_vals[start..].iter()) {
        if fn_key(a) == fn_key(b) {
            take_a!(start, start);
            start += 1;
        } else {
            break;
        }
    }

    // Skip common suffix.
    for (a, b) in a_vals[start..]
        .iter()
        .rev()
        .zip(b_vals[start..].iter().rev())
    {
        if fn_key(a) == fn_key(b) {
            a_end -= 1;
            b_end -= 1;
            take_a!(a_end, b_end);
        } else {
            break;
        }
    }

    if start == a_end {
        // Insert new views.
        let next = get_next_b!(b_end - 1).cloned();
        for i in start..b_end {
            new_b!(i);
            insert_b!(i, next.as_ref());
        }
    } else if start == b_end {
        // Remove rest views.
        for i in start..a_end {
            remove_a!(i);
        }
    } else {
        let len = b_end - start;
        let mut sources = vec![Source::New; len];
        let key_index = b_vals[start..b_end]
            .iter()
            .enumerate()
            .map(|(i, b)| (fn_key(b), start + i))
            .collect::<HashMap<_, _>>();

        let mut should_move = false;
        let mut pos = 0;
        for (i, a) in a_vals[start..a_end].iter().enumerate() {
            let i = start + i;
            if let Some(&j) = key_index.get(&fn_key(a)) {
                if !should_move {
                    if pos < j {
                        pos = j;
                    } else {
                        should_move = true;
                    }
                }
                sources[j - start] = Source::Move(i);
                take_a!(i, j);
            } else {
                remove_a!(i);
            }
        }

        if should_move {
            mark_lis(&mut sources);
        }
        debug!("sources({start}, {b_end}): {sources:?}");
        for (i, &s) in sources.iter().enumerate().rev() {
            let i = start + i;
            match s {
                Source::New => {
                    new_b!(i);
                    insert_b!(i);
                }
                Source::Move(_) if should_move => insert_b!(i),
                _ => {}
            }
        }
    }

    if is_debug!() {
        for view in b_views.iter() {
            assert!(!dummy_view.ref_eq(view));
        }
        for disposer in a_disposers.iter() {
            assert!(disposer.is_none());
        }
    }
    (b_views, b_disposers)
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Source {
    Lis,
    New,
    Move(usize),
}

// Modified from: https://github.com/localvoid/ivi/blob/a769a11/packages/ivi/src/vdom/reconciler.ts#L960
fn mark_lis(sources: &mut [Source]) {
    let mut predecessors = vec![0; sources.len()];
    let mut starts = vec![0; sources.len()];

    let first = sources
        .iter()
        .position(|s| matches!(s, Source::Move(..)))
        .unwrap_or(0);
    starts[0] = first;

    let mut l = 0;
    for (i, &k) in sources[first..].iter().enumerate() {
        if k == Source::New {
            continue;
        }
        let i = first + i;
        let j = starts[l];
        if sources[j] < k {
            predecessors[i] = j;
            l += 1;
            starts[l] = i;
        } else {
            let mut lo = 0;
            let mut hi = l;

            while lo < hi {
                let mid = (lo >> 1) + (hi >> 1) + (lo & hi & 1);
                if sources[starts[mid]] < k {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }

            if k < sources[starts[lo]] {
                if lo > 0 {
                    predecessors[i] = starts[lo - 1];
                }
                starts[lo] = i;
            }
        }
    }

    let mut i = starts[l];
    for _ in 0..=l {
        sources[i] = Source::Lis;
        i = predecessors[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lis() {
        use Source::*;

        let mut sources = vec![
            New,
            Move(2),
            Move(8),
            New,
            Move(9),
            Move(5),
            Move(6),
            New,
            Move(7),
            Move(1),
            New,
        ];
        mark_lis(&mut sources);
        let expected = vec![
            New,
            Lis,
            Move(8),
            New,
            Move(9),
            Lis,
            Lis,
            New,
            Lis,
            Move(1),
            New,
        ];
        assert_eq!(sources, expected);
    }
}
