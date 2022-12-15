use std::marker::PhantomData;
use xframe_core::{
    component::{ComponentInit, ComponentNode, ComponentRender, GenericComponent},
    Attribute, Component, GenericElement, GenericNode, IntoReactive,
};
use xframe_reactive::Scope;

macro_rules! Element {
    ($N:ident, $Identifier:ty) => {
        Element<
            $N,
            impl ElementInit<$N>,
            impl ElementRender<$N>,
            $Identifier,
        >
    };
}

pub fn view<N, E>(cx: Scope, render: impl 'static + FnOnce(E)) -> Element!(N, E)
where
    N: GenericNode,
    E: GenericElement<N>,
{
    Element(cx).root(render)
}

#[allow(non_snake_case)]
pub fn Element<N: GenericNode>(cx: Scope) -> Element<N, (), (), ()> {
    Element {
        cx,
        init: (),
        render: (),
        node: PhantomData,
        identifier: PhantomData,
    }
}

pub struct Element<N, Init, Render, Identifier> {
    cx: Scope,
    init: Init,
    render: Render,
    node: PhantomData<N>,
    identifier: PhantomData<Identifier>,
}

impl<N: GenericNode> Element<N, (), (), ()> {
    pub fn root<E>(self, render: impl 'static + FnOnce(E)) -> Element!(N, E)
    where
        E: GenericElement<N>,
    {
        let cx = self.cx;
        Element {
            cx,
            init: ElementInitImpl(move || N::create(E::TYPE)),
            render: ElementRenderImpl(move |node: N| {
                let next_sibling = node.next_sibling();
                let last_child = node.first_child();
                render(E::create_with_node(cx, node));
                (next_sibling, last_child)
            }),
            node: PhantomData,
            identifier: PhantomData,
        }
    }
}

impl<N, Init, Render, Identifier> Element<N, Init, Render, Identifier>
where
    N: GenericNode,
    Init: ElementInit<N>,
    Render: ElementRender<N>,
    Identifier: 'static,
{
    pub fn build(self) -> impl GenericComponent<N> {
        self
    }

    pub fn child<C>(self, component: C) -> Element!(N, (Identifier, C::Identifier))
    where
        C: GenericComponent<N>,
    {
        let component = component.into_component_node();
        Element {
            cx: self.cx,
            init: ElementInitImpl(move || {
                let root = self.init.init_element();
                component.init.init().append_to(&root);
                root
            }),
            render: ElementRenderImpl(move |node: N| {
                let (root_sibling, mut last_child) = self.render.render_element(node);
                last_child = component
                    .render
                    .render(last_child.unwrap_or_else(|| unreachable!()));
                (root_sibling, last_child)
            }),
            node: PhantomData,
            identifier: PhantomData,
        }
    }

    pub fn child_element<E, F>(self, render: F) -> Element!(N, (Identifier, E))
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
    ) -> Element!(N, (Identifier, crate::elements::text<N>)) {
        let data = data.into_reactive();
        self.child_element(move |text: crate::elements::text<_>| {
            text.data(data);
        })
    }
}

impl<N, Init, Render, Identifier> From<Element<N, Init, Render, Identifier>>
    for ComponentNode<Init, Render>
{
    fn from(t: Element<N, Init, Render, Identifier>) -> Self {
        Self {
            init: t.init,
            render: t.render,
        }
    }
}

impl<N, Init, Render, Identifier> GenericComponent<N> for Element<N, Init, Render, Identifier>
where
    N: GenericNode,
    Init: ElementInit<N>,
    Render: ElementRender<N>,
    Identifier: 'static,
{
    type Init = Init;
    type Render = Render;
    type Identifier = Identifier;
}

pub trait ElementInit<N: GenericNode>: ComponentInit<N> {
    /// Initialize and return the root node of this element.
    fn init_element(self) -> N;
}

struct ElementInitImpl<F>(F);

impl<N, F> ElementInit<N> for ElementInitImpl<F>
where
    N: GenericNode,
    F: 'static + FnOnce() -> N,
{
    fn init_element(self) -> N {
        (self.0)()
    }
}

impl<N, F> ComponentInit<N> for ElementInitImpl<F>
where
    N: GenericNode,
    ElementInitImpl<F>: ElementInit<N>,
{
    fn init(self) -> Component<N> {
        Component::Node(self.init_element())
    }
}

pub trait ElementRender<N: GenericNode>: ComponentRender<N> {
    /// Render and return the next sibling and the last child of root node.
    fn render_element(self, node: N) -> (Option<N>, Option<N>);
}

struct ElementRenderImpl<F>(F);

impl<N, F> ElementRender<N> for ElementRenderImpl<F>
where
    N: GenericNode,
    F: 'static + FnOnce(N) -> (Option<N>, Option<N>),
{
    fn render_element(self, node: N) -> (Option<N>, Option<N>) {
        (self.0)(node)
    }
}

impl<N, F> ComponentRender<N> for ElementRenderImpl<F>
where
    N: GenericNode,
    ElementRenderImpl<F>: ElementRender<N>,
{
    fn render(self, node: N) -> Option<N> {
        let (next_sibling, last_child) = self.render_element(node);
        debug_assert!(last_child.is_none());
        next_sibling
    }
}
