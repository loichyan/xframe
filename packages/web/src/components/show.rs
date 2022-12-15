use crate::view;
use smallvec::SmallVec;
use std::marker::PhantomData;
use xframe_core::{
    component::DynComponent, Component, GenericComponent, GenericElement, GenericNode,
    IntoReactive, Reactive, Value,
};
use xframe_reactive::Scope;

const INITIAL_BRANCH_SLOTS: usize = 2;

#[allow(non_snake_case)]
pub fn Show<N: GenericNode>(cx: Scope) -> Show<N> {
    Show {
        cx,
        branches: Default::default(),
        default: None,
    }
}

pub struct Show<N> {
    cx: Scope,
    branches: SmallVec<[If<N>; INITIAL_BRANCH_SLOTS]>,
    default: Option<Else<N>>,
}

pub enum ShowChild<N> {
    If(If<N>),
    Else(Else<N>),
}

impl<N> From<If<N>> for ShowChild<N> {
    fn from(t: If<N>) -> Self {
        Self::If(t)
    }
}

impl<N> From<Else<N>> for ShowChild<N> {
    fn from(t: Else<N>) -> Self {
        Self::Else(t)
    }
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

        let Self {
            cx,
            branches,
            default,
        } = self;
        view(cx, move |placeholder: crate::elements::placeholder<N>| {
            let placeholder = placeholder
                .desc("placeholder for the `xframe::Show` component")
                .into_node();
            let parent = placeholder.parent().unwrap_or_else(|| unreachable!());
            let mut current = Component::Node(placeholder.clone());
            let branches = branches
                .into_iter()
                .map(|If::<N> { when, children, .. }| Branch {
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
                    children: default
                        .map(|default| default.children.render())
                        .unwrap_or_else(|| current.clone()),
                }))
                .collect::<SmallVec<[Branch<N>; INITIAL_BRANCH_SLOTS]>>();
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
    pub fn child<C: Into<ShowChild<N>>>(mut self, child: C) -> Show<N> {
        match child.into() {
            ShowChild::If(case) => self.branches.push(case),
            ShowChild::Else(default) => self.default = Some(default),
        }
        self
    }
}

#[allow(non_snake_case)]
pub fn If<N: GenericNode>(_: Scope) -> If<N, (), ()> {
    If {
        marker: PhantomData,
        when: (),
        children: (),
    }
}

pub struct If<N, When = Reactive<bool>, Children = DynComponent<N>> {
    when: When,
    children: Children,
    marker: PhantomData<N>,
}

impl<N: GenericNode> If<N> {
    pub fn build(self) -> Self {
        self
    }
}

impl<N, Children> If<N, (), Children>
where
    N: GenericNode,
{
    pub fn when<T: IntoReactive<bool>>(self, when: T) -> If<N, Reactive<bool>, Children> {
        let Self {
            children: child,
            marker,
            ..
        } = self;
        If {
            when: when.into_reactive(),
            children: child,
            marker,
        }
    }
}

impl<N, When> If<N, When, ()>
where
    N: GenericNode,
{
    pub fn child<C: GenericComponent<N>>(self, child: C) -> If<N, When, DynComponent<N>> {
        let Self { when, marker, .. } = self;
        If {
            when,
            children: child.into_dyn_component(),
            marker,
        }
    }
}

#[allow(non_snake_case)]
pub fn Else<N: GenericNode>(_: Scope) -> Else<N, ()> {
    Else {
        children: (),
        marker: PhantomData,
    }
}

pub struct Else<N, Children = DynComponent<N>> {
    children: Children,
    marker: PhantomData<N>,
}

impl<N: GenericNode> Else<N> {
    pub fn build(self) -> Self {
        self
    }
}

impl<N> Else<N, ()>
where
    N: GenericNode,
{
    pub fn child<C: GenericComponent<N>>(self, child: C) -> Else<N, DynComponent<N>> {
        let Self { marker, .. } = self;
        Else {
            children: child.into_dyn_component(),
            marker,
        }
    }
}
