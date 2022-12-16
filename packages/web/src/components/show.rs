use crate::view;
use smallvec::SmallVec;
use xframe_core::{
    component::DynComponent, Component, GenericComponent, GenericElement, GenericNode,
    IntoReactive, Reactive, Value,
};
use xframe_reactive::Scope;

const INITIAL_BRANCH_SLOTS: usize = 4;

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
    children: DynComponent<N>,
}

impl<N> Show<N>
where
    N: GenericNode,
{
    pub fn build(self) -> impl GenericComponent<N> {
        struct Branch<N> {
            when: Reactive<bool>,
            children: Component<N>,
        }

        let Self { cx, branches } = self;
        view(cx, move |placeholder: crate::elements::placeholder<N>| {
            let placeholder = placeholder
                .desc("placeholder for the `xframe::Show` component")
                .into_node();
            let parent = placeholder.parent().unwrap_or_else(|| unreachable!());
            let mut current = Component::Node(placeholder.clone());
            let branches = branches
                .into_iter()
                .map(|ShowChild { when, children }| Branch {
                    when,
                    children: {
                        // Replace empty fragments with the placeholder.
                        let mut children = children.render();
                        children.insert_with(|| placeholder.clone());
                        children
                    },
                })
                // Add a default branch.
                .chain(std::iter::once(Branch {
                    when: Value(true),
                    children: current.clone(),
                }))
                .collect::<SmallVec<[_; INITIAL_BRANCH_SLOTS]>>();
            cx.create_effect(move || {
                for branch in branches.iter() {
                    let branch: &Branch<N> = branch;
                    if branch.when.clone().into_value() {
                        let old_node = current.first().unwrap_or_else(|| unreachable!());
                        let new_node = branch.children.first().unwrap_or_else(|| unreachable!());
                        if old_node.ne(new_node) {
                            current.replace_with(&parent, &branch.children);
                            current = branch.children.clone();
                        }
                        break;
                    }
                }
            });
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
            children: self.children.expect("no `child` was provided"),
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
            children: self.children.expect("no `child` was provided"),
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
