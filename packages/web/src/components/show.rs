use crate::view;
use smallvec::SmallVec;
use std::marker::PhantomData;
use xframe_core::{
    Component, GenericComponent, GenericElement, GenericNode, IntoReactive, Reactive, Value,
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
        let Self {
            cx,
            mut branches,
            default,
        } = self;
        if let Some(default) = default {
            branches.push(If {
                when: Value(true),
                children: default.children,
                marker: PhantomData,
            });
        }
        view(cx, move |placeholder: crate::elements::placeholder<N>| {
            let placeholder = placeholder.into_node();
            let parent = placeholder.parent().unwrap_or_else(|| unreachable!());
            let mut current = Component::Node(placeholder.clone());
            cx.create_effect(move || {
                for branch in branches.iter() {
                    // TODO: figure out why rust-analyzer cannot infer the type of branch
                    let branch: &If<N> = branch;
                    if branch.when.clone().into_value() {
                        let old_node = current.first().unwrap_or(&placeholder);
                        let new_node = branch.children.first().unwrap_or(&placeholder);
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

pub struct If<N, When = Reactive<bool>, Children = Component<N>> {
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
    pub fn child<C: GenericComponent<N>>(self, child: C) -> If<N, When, Component<N>> {
        let Self { when, marker, .. } = self;
        If {
            when,
            children: child.render(),
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

pub struct Else<N, Children = Component<N>> {
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
    pub fn child<C: GenericComponent<N>>(self, child: C) -> Else<N, Component<N>> {
        let Self { marker, .. } = self;
        Else {
            children: child.render(),
            marker,
        }
    }
}
