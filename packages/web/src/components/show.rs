use crate::view_with;
use smallvec::SmallVec;
use xframe_core::{
    component::DynComponent, GenericComponent, GenericElement, GenericNode, IntoReactive, Reactive,
    Value, View,
};
use xframe_reactive::Scope;

const INITIAL_BRANCH_SLOTS: usize = 4;

define_placeholder!(Placeholder("Placeholder for `xframe::Show` Component"));

#[allow(non_snake_case)]
pub fn Show<N: GenericNode>(cx: Scope) -> Show<N> {
    Show {
        cx,
        children: Default::default(),
    }
}

pub struct Show<N> {
    cx: Scope,
    children: SmallVec<[ShowChild<N>; INITIAL_BRANCH_SLOTS]>,
}

pub struct ShowChild<N> {
    cond: Reactive<bool>,
    content: DynComponent<N>,
}

struct Branch<N> {
    cond: Reactive<bool>,
    view: View<N>,
}

impl<N> Show<N>
where
    N: GenericNode,
{
    pub fn build(self) -> impl GenericComponent<N> {
        let Self { cx, children } = self;
        view_with(cx, move |placeholder: Placeholder<N>| {
            let placeholder = placeholder.into_node();
            // Add a default branch.
            let branches = children
                .into_iter()
                .map(|ShowChild { cond, content }| Branch {
                    cond,
                    view: cx.untrack(|| content.render()),
                })
                .chain(Some(Branch {
                    cond: Value(true),
                    view: View::Node(placeholder.clone()),
                }))
                .collect::<SmallVec<[_; INITIAL_BRANCH_SLOTS]>>();
            let dyn_view = cx.create_signal(View::Node(placeholder));
            cx.create_effect(move || {
                for branch in branches.iter() {
                    let Branch::<N> {
                        cond,
                        view: new_view,
                    } = branch;
                    if cond.clone().into_value() {
                        cx.untrack(|| {
                            let current = dyn_view.get();
                            let old_node = current.first();
                            let parent = old_node.parent().unwrap_or_else(|| unreachable!());
                            let new_node = new_view.first();
                            if old_node.ne(&new_node) {
                                current.replace_with(&parent, new_view);
                                dyn_view.set(new_view.clone());
                            }
                        });
                        break;
                    }
                }
            });
            View::from(dyn_view)
        })
    }
}

impl<N> Show<N> {
    pub fn child(mut self, child: ShowChild<N>) -> Show<N> {
        self.children.push(child);
        self
    }
}

#[allow(non_snake_case)]
pub fn If<N: GenericNode>(_: Scope) -> If<N> {
    If {
        when: None,
        children: None,
    }
}

pub struct If<N> {
    when: Option<Reactive<bool>>,
    children: Option<DynComponent<N>>,
}

impl<N: GenericNode> If<N> {
    pub fn build(self) -> ShowChild<N> {
        ShowChild {
            cond: self.when.expect("no `when` was provided"),
            content: self.children.expect("no `child` was provided"),
        }
    }
}

impl<N: GenericNode> If<N> {
    pub fn when<T: IntoReactive<bool>>(mut self, when: T) -> If<N> {
        if self.when.is_some() {
            panic!("`when` has been already provided");
        }
        self.when = Some(when.into_reactive());
        self
    }

    pub fn child<C: GenericComponent<N>>(mut self, child: C) -> If<N> {
        if self.children.is_some() {
            panic!("only one `child` should be provided");
        }
        self.children = Some(child.into_dyn_component());
        self
    }
}

#[allow(non_snake_case)]
pub fn Else<N: GenericNode>(_: Scope) -> Else<N> {
    Else { children: None }
}

pub struct Else<N> {
    children: Option<DynComponent<N>>,
}

impl<N: GenericNode> Else<N> {
    pub fn build(self) -> ShowChild<N> {
        ShowChild {
            cond: Value(true),
            content: self.children.expect("no `child` was provided"),
        }
    }
}

impl<N: GenericNode> Else<N> {
    pub fn child<C: GenericComponent<N>>(mut self, child: C) -> Else<N> {
        if self.children.is_some() {
            panic!("only one `child` should be provided");
        }
        self.children = Some(child.into_dyn_component());
        self
    }
}
