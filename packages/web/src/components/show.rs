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
        branches: Default::default(),
    }
}

pub struct Show<N> {
    cx: Scope,
    branches: SmallVec<[ShowChild<N>; INITIAL_BRANCH_SLOTS]>,
}

pub struct ShowChild<N> {
    when: Reactive<bool>,
    children: View<N>,
}

impl<N> Show<N>
where
    N: GenericNode,
{
    pub fn build(self) -> impl GenericComponent<N> {
        let Self { cx, mut branches } = self;
        view_with(cx, move |placeholder: Placeholder<N>| {
            let placeholder = placeholder.into_node();
            let parent = placeholder.parent().unwrap_or_else(|| unreachable!());
            // Add a default branch.
            branches.push(ShowChild {
                when: Value(true),
                children: View::Node(placeholder.clone()),
            });
            let dyn_view = cx.create_signal(View::Node(placeholder));
            cx.create_effect(move || {
                for branch in branches.iter() {
                    let ShowChild::<N> {
                        when,
                        children: new_view,
                    } = branch;
                    if when.clone().into_value() {
                        cx.untrack(|| {
                            let current = dyn_view.get();
                            let old_node = current.first();
                            let new_node = new_view.first();
                            if old_node.ne(&new_node) {
                                cx.untrack(|| {
                                    current.replace_with(&parent, new_view);
                                });
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
        self.branches.push(child);
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
            when: self.when.expect("no `when` was provided"),
            children: self.children.expect("no `child` was provided").render(),
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
            when: Value(true),
            children: self.children.expect("no `child` was provided").render(),
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
