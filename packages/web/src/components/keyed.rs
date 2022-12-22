use crate::{utils::Visit, view_with};
use std::hash::Hash;
use xframe_core::{GenericComponent, GenericElement, GenericNode, IntoReactive, Reactive, View};
use xframe_reactive::{untrack, Scope};

type AIndexMap<K, V> = indexmap::IndexMap<K, V, ahash::RandomState>;

define_placeholder!(Placeholder("Placeholder for `xframe::Keyed` Component"));

pub struct Keyed<N, T, K, I>
where
    I: 'static + Visit<T>,
{
    cx: Scope,
    each: Option<Reactive<I>>,
    key: Option<Box<dyn Fn(&T) -> K>>,
    children: Option<Box<dyn Fn(&T) -> View<N>>>,
}

// TODO: rename to `For`
#[allow(non_snake_case)]
pub fn Keyed<N, T, K, I>(cx: Scope) -> Keyed<N, T, K, I>
where
    N: GenericNode,
    I: 'static + Clone + Visit<T>,
{
    Keyed {
        cx,
        each: None,
        key: None,
        children: None,
    }
}

impl<N, T, K, I> Keyed<N, T, K, I>
where
    N: GenericNode,
    T: 'static,
    K: 'static + Eq + Hash,
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
        let each = each.expect("`each` was not provided");
        let fn_key = key.expect("`key` was not provided");
        let fn_view = children.expect("`each` was not provided");
        view_with(cx, move |placeholder: Placeholder<N>| {
            let placeholder = placeholder.into_node();
            let mut mounted_views = AIndexMap::<K, View<N>>::default();
            let dyn_view = cx.create_signal(View::Node(placeholder.clone()));
            cx.create_effect(move || {
                let each = each.clone().into_value();
                untrack(|| {
                    let mut new_mounted_views = AIndexMap::default();
                    each.visit(|val| {
                        let key = fn_key(val);
                        if let Some(view) = mounted_views.get(&key) {
                            new_mounted_views.insert(key, view.clone());
                        } else {
                            let view = fn_view(val);
                            new_mounted_views.insert(key, view);
                        }
                    });
                    let new_view = new_mounted_views.values().cloned().collect::<Vec<_>>();
                    let mounted_views = std::mem::replace(&mut mounted_views, new_mounted_views);
                    let current_view = dyn_view.get();
                    let current_last = current_view.last();
                    let parent = current_last.parent().unwrap();
                    let next_last = current_last.next_sibling();
                    if new_view.is_empty() {
                        if current_last.ne(&placeholder) {
                            let placeholder = View::Node(placeholder.clone());
                            current_view.replace_with(&parent, &placeholder);
                            dyn_view.set(placeholder);
                        }
                    } else {
                        if new_view.len() > mounted_views.len() && current_last.eq(&placeholder) {
                            parent.remove_child(&placeholder);
                        }
                        // TODO: do a diff?
                        for view in mounted_views.into_values() {
                            view.remove_from(&parent);
                        }
                        let new_view = View::from(new_view);
                        new_view.move_before(&parent, next_last.as_ref());
                        dyn_view.set(new_view);
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

    pub fn key(mut self, key: impl 'static + Fn(&T) -> K) -> Self {
        if self.key.is_some() {
            panic!("`child` has already been provided");
        }
        self.key = Some(Box::new(key));
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
