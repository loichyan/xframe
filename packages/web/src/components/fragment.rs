use crate::view;
use std::{marker::PhantomData, rc::Rc};
use xframe_core::{
    component::{ComponentInit, ComponentNode, ComponentRender},
    Attribute, Component, GenericComponent, GenericElement, GenericNode, IntoReactive,
};
use xframe_reactive::Scope;

macro_rules! Fragment {
    ($N:ident, $Identifier:ty) => {
        Fragment<
            $N,
            impl FragmentInit<$N>,
            impl FragmentRender<$N>,
            $Identifier,
        >
    };
}

#[allow(non_snake_case)]
pub fn Fragment<N: GenericNode>(cx: Scope) -> Fragment!(N, ()) {
    Fragment {
        cx,
        init: FragmentInitImpl(move || Vec::new()),
        render: FragmentRenderImpl(move |node: Option<N>| node),
        identifier: PhantomData,
        marker: PhantomData,
    }
}

pub struct Fragment<N, Init, Render, Identifier> {
    cx: Scope,
    init: Init,
    render: Render,
    identifier: PhantomData<Identifier>,
    marker: PhantomData<N>,
}

impl<N, Init, Render, Identifier> Fragment<N, Init, Render, Identifier>
where
    N: GenericNode,
    Init: FragmentInit<N>,
    Render: FragmentRender<N>,
    Identifier: 'static,
{
    pub fn build(self) -> impl GenericComponent<N> {
        self
    }

    pub fn child<C>(self, child: C) -> Fragment!(N, (Identifier, C::Identifier))
    where
        C: GenericComponent<N>,
    {
        let child = child.into_component_node();
        Fragment {
            cx: self.cx,
            init: FragmentInitImpl(move || {
                let mut fragment = self.init.init_fragment();
                fragment.extend(child.init.init().iter().cloned());
                fragment
            }),
            render: FragmentRenderImpl(move |node: Option<N>| {
                let next_sibling = self.render.render_fragment(node);
                child.render.render(next_sibling)
            }),
            identifier: PhantomData,
            marker: PhantomData,
        }
    }

    pub fn child_element<E, F>(self, render: F) -> Fragment!(N, (Identifier, E))
    where
        E: GenericElement<N>,
        F: 'static + FnOnce(E),
    {
        let cx = self.cx;
        self.child(view(cx, render))
    }

    pub fn child_text<A: IntoReactive<Attribute>>(
        self,
        data: A,
    ) -> Fragment!(N, (Identifier, crate::elements::text<N>)) {
        let data = data.into_reactive();
        self.child_element(move |text: crate::elements::text<_>| {
            text.data(data);
        })
    }
}

impl<N, Init, Render, Identifier> From<Fragment<N, Init, Render, Identifier>>
    for ComponentNode<Init, Render>
{
    fn from(t: Fragment<N, Init, Render, Identifier>) -> Self {
        Self {
            init: t.init,
            render: t.render,
        }
    }
}

impl<N, Init, Render, Identifier> GenericComponent<N> for Fragment<N, Init, Render, Identifier>
where
    N: GenericNode,
    Init: FragmentInit<N>,
    Render: FragmentRender<N>,
    Identifier: 'static,
{
    type Init = Init;
    type Render = Render;
    type Identifier = Identifier;
}

pub trait FragmentInit<N: GenericNode>: ComponentInit<N> {
    /// Initialize and return the root node of this element.
    fn init_fragment(self) -> Vec<N>;
}

struct FragmentInitImpl<F>(F);

impl<N, F> FragmentInit<N> for FragmentInitImpl<F>
where
    N: GenericNode,
    F: 'static + FnOnce() -> Vec<N>,
{
    fn init_fragment(self) -> Vec<N> {
        (self.0)()
    }
}

impl<N, F> ComponentInit<N> for FragmentInitImpl<F>
where
    N: GenericNode,
    FragmentInitImpl<F>: FragmentInit<N>,
{
    fn init(self) -> Component<N> {
        Component::Fragment(Rc::from(self.init_fragment().into_boxed_slice()))
    }
}

pub trait FragmentRender<N: GenericNode>: ComponentRender<N> {
    /// Render and return the next sibling and the last child of root node.
    fn render_fragment(self, node: Option<N>) -> Option<N>;
}

struct FragmentRenderImpl<F>(F);

impl<N, F> FragmentRender<N> for FragmentRenderImpl<F>
where
    N: GenericNode,
    F: 'static + FnOnce(Option<N>) -> Option<N>,
{
    fn render_fragment(self, node: Option<N>) -> Option<N> {
        (self.0)(node)
    }
}

impl<N, F> ComponentRender<N> for FragmentRenderImpl<F>
where
    N: GenericNode,
    FragmentRenderImpl<F>: FragmentRender<N>,
{
    fn render(self, node: Option<N>) -> Option<N> {
        self.render_fragment(node)
    }
}
