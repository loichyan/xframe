use crate::element::GenericElement;
use std::rc::Rc;
use xframe_core::{
    view::ViewParentExt, GenericComponent, GenericNode, IntoReactive, Reactive, RenderInput,
    RenderOutput, View,
};
use xframe_reactive::{untrack, Scope, ScopeDisposer};

define_placeholder!(struct Placeholder("PLACEHOLDER FOR `xframe::List` COMPONENT"));

pub struct List<N, T> {
    input: RenderInput<N>,
    each: Option<Reactive<Vec<T>>>,
    children: Option<Box<dyn Fn(Scope, &T) -> View<N>>>,
}

#[allow(non_snake_case)]
pub fn List<N, T>(cx: Scope) -> List<N, T>
where
    N: GenericNode,
    T: 'static + Clone,
{
    GenericComponent::new(cx)
}

impl<N, T> GenericComponent<N> for List<N, T>
where
    N: GenericNode,
    T: 'static + Clone,
{
    fn new_with_input(input: RenderInput<N>) -> Self {
        Self {
            input,
            each: None,
            children: None,
        }
    }

    fn render_to_output(self) -> RenderOutput<N> {
        let Self {
            input,
            each,
            children,
        } = self;
        let cx = input.cx;
        let each = each.expect("`List::each` was not specified");
        let fn_view = children.expect("`List::child` was not specified");

        let mut current_fragment = Rc::new([]) as Rc<[View<N>]>;
        let mut current_disposers = Vec::<ScopeDisposer>::new();
        let mut placeholder = None;
        Placeholder::new_with_input(input)
            .dyn_view(move |current_view| {
                let placeholder = &*placeholder.get_or_insert_with(|| current_view.clone());
                // Only `each` needs to be tracked.
                let new_vals = each.clone().into_value();
                untrack(|| {
                    let parent = current_view.parent();
                    let current_len = current_fragment.len();
                    let new_len = new_vals.len();
                    let new_view: View<N>;
                    if new_len == 0 {
                        if current_len == 0 {
                            return current_view;
                        }
                        // Replace empty view with a placeholder.
                        parent.replace_child(placeholder, &current_view);
                        current_fragment = Rc::new([]);
                        current_disposers = Vec::new();
                        new_view = placeholder.clone();
                    } else if new_len < current_len {
                        let (lhs, rhs) = current_fragment.split_at(new_len);
                        // Remove extra views.
                        for view in rhs {
                            parent.remove_child(view);
                        }
                        current_fragment = lhs.to_vec().into_boxed_slice().into();
                        current_disposers.truncate(new_len);
                        new_view = View::fragment_shared(current_fragment.clone());
                    } else if new_len > current_len {
                        let next_sibling = current_view.next_sibling();
                        let mut new_fragment = current_fragment.to_vec();
                        for val in new_vals[current_len..].iter() {
                            // Append new views.
                            let (view, disposer) = cx.create_child(|cx| fn_view(cx, val));
                            parent.insert_before(&view, next_sibling.as_ref());
                            new_fragment.push(view);
                            current_disposers.push(disposer);
                        }
                        if current_len == 0 {
                            // Remove the placeholder.
                            parent.remove_child(placeholder);
                        }
                        current_fragment = new_fragment.into_boxed_slice().into();
                        new_view = View::fragment_shared(current_fragment.clone());
                    } else {
                        return current_view;
                    }
                    debug_assert_eq!(current_fragment.len(), current_disposers.len());
                    debug_assert!(new_view.check_mount_order());
                    new_view
                })
            })
            .render_to_output()
    }
}

impl<N, T> List<N, T>
where
    N: GenericNode,
    T: 'static + Clone,
{
    pub fn each<E: IntoReactive<Vec<T>>>(mut self, each: E) -> Self {
        if self.each.is_some() {
            panic!("`List::each` has been specified");
        }
        self.each = Some(each.into_reactive(self.input.cx));
        self
    }

    pub fn child<C: GenericComponent<N>>(
        mut self,
        child: impl 'static + Fn(Scope, &T) -> C,
    ) -> Self {
        if self.children.is_some() {
            panic!("`List::child` has been specified");
        }
        self.children = Some(Box::new(move |cx, val| child(cx, val).render()));
        self
    }
}
