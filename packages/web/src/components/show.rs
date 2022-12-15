use crate::create_component;
use smallvec::SmallVec;
use std::marker::PhantomData;
use wasm_bindgen::UnwrapThrowExt;
use xframe_core::{GenericComponent, GenericElement, GenericNode, IntoReactive, Reactive};
use xframe_reactive::Scope;

const INITIAL_BRANCH_SLOTS: usize = 2;

#[allow(non_snake_case)]
pub fn Show<N: GenericNode>(cx: Scope) -> Show<N> {
    Show {
        cx,
        branches: Default::default(),
        default: None,
        marker: PhantomData,
    }
}

pub struct Show<N> {
    cx: Scope,
    branches: SmallVec<[If<N>; INITIAL_BRANCH_SLOTS]>,
    default: Option<Else<N>>,
    marker: PhantomData<N>,
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
            branches,
            default,
            ..
        } = self;
        create_component(cx, move |placeholder: crate::elements::placeholder<N>| {
            let mut current = placeholder.into_node();
            let parent = current
                .parent()
                .expect_throw("`Show` component must have a parent");
            cx.create_effect(move || {
                for branch in branches.iter() {
                    if branch.when.clone().into_value() && current.ne(&branch.child) {
                        parent.replace_child(&branch.child, &current);
                        current = branch.child.clone();
                        return;
                    }
                }
                if let Some(default) = default.as_ref() {
                    if current.ne(&default.child) {
                        parent.replace_child(&default.child, &current);
                        current = default.child.clone();
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
        child: (),
    }
}

pub struct If<N, When = Reactive<bool>, Child = N> {
    when: When,
    child: Child,
    marker: PhantomData<N>,
}

impl<N: GenericNode> If<N> {
    pub fn build(self) -> Self {
        self
    }
}

impl<N, When, Child> If<N, When, Child>
where
    N: GenericNode,
{
    pub fn when<T: IntoReactive<bool>>(self, when: T) -> If<N, Reactive<bool>, Child> {
        let Self { child, marker, .. } = self;
        If {
            when: when.into_reactive(),
            child,
            marker,
        }
    }

    pub fn child<C: GenericComponent<N>>(self, child: C) -> If<N, When, N> {
        let Self { when, marker, .. } = self;
        If {
            when,
            child: child.render(),
            marker,
        }
    }
}

#[allow(non_snake_case)]
pub fn Else<N: GenericNode>(_: Scope) -> Else<N, ()> {
    Else {
        child: (),
        marker: PhantomData,
    }
}

pub struct Else<N, Child = N> {
    child: Child,
    marker: PhantomData<N>,
}

impl<N: GenericNode> Else<N> {
    pub fn build(self) -> Self {
        self
    }
}

impl<N, Child> Else<N, Child>
where
    N: GenericNode,
{
    pub fn child<C: GenericComponent<N>>(self, child: C) -> Else<N, N> {
        let Self { marker, .. } = self;
        Else {
            child: child.render(),
            marker,
        }
    }
}
