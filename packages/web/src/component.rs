use std::marker::PhantomData;
use xframe_core::{
    component::{ComponentInit, ComponentNode, ComponentRender, GenericComponent},
    Attribute, GenericElement, GenericNode, IntoReactive,
};
use xframe_reactive::Scope;

pub fn create_component<N, E>(
    cx: Scope,
    render: impl 'static + FnOnce(E),
) -> Component<N, impl InitChain<N>, impl RenderChain<N>, E>
where
    N: GenericNode,
    E: GenericElement<N>,
{
    Component {
        cx,
        init: InitImpl(PhantomData, move || N::create(E::TYPE)),
        render: RenderImpl(PhantomData, move |root: N| {
            let next = root.first_child();
            render(E::create_with_node(cx, root));
            next
        }),
        node: PhantomData,
        identifier: PhantomData,
    }
}

pub struct Component<N, Init, Render, Identifier> {
    cx: Scope,
    init: Init,
    render: Render,
    node: PhantomData<N>,
    identifier: PhantomData<Identifier>,
}

impl<N, Init, Render, Identifier> Component<N, Init, Render, Identifier>
where
    N: GenericNode,
    Init: InitChain<N>,
    Render: RenderChain<N>,
    Identifier: 'static,
{
    pub fn child<C>(
        self,
        component: C,
    ) -> Component<N, impl InitChain<N>, impl RenderChain<N>, (Identifier, C::Identifier)>
    where
        C: GenericComponent<N>,
    {
        let component = component.into_component_node();
        Component {
            cx: self.cx,
            init: InitImpl(PhantomData, move || {
                let root = self.init.init_and_return_root();
                root.append_child(&component.init.init());
                root
            }),
            render: RenderImpl(PhantomData, move |prev: N| {
                let node = self
                    .render
                    .render_and_return_next(prev)
                    .unwrap_or_else(|| unreachable!());
                let next = node.next_sibling();
                component.render.render(node);
                next
            }),
            node: PhantomData,
            identifier: PhantomData,
        }
    }

    pub fn child_element<E, F>(
        self,
        render: F,
    ) -> Component<N, impl InitChain<N>, impl RenderChain<N>, (Identifier, E)>
    where
        E: GenericElement<N>,
        F: 'static + FnOnce(E),
    {
        let cx = self.cx;
        self.child(create_component(cx, render))
    }

    pub fn child_text<A: IntoReactive<Attribute>>(
        self,
        data: A,
    ) -> Component<N, impl InitChain<N>, impl RenderChain<N>, (Identifier, crate::elements::text<N>)>
    {
        let data = data.into_reactive();
        self.child_element(move |text: crate::elements::text<_>| {
            text.data(data);
        })
    }
}

impl<N, Init, Render, Identifier> From<Component<N, Init, Render, Identifier>>
    for ComponentNode<Init, Render>
{
    fn from(t: Component<N, Init, Render, Identifier>) -> Self {
        Self {
            init: t.init,
            render: t.render,
        }
    }
}

impl<N, Init, Render, Identifier> GenericComponent<N> for Component<N, Init, Render, Identifier>
where
    N: GenericNode,
    Init: ComponentInit<N>,
    Render: ComponentRender<N>,
    Identifier: 'static,
{
    type Init = Init;
    type Render = Render;
    type Identifier = Identifier;
}

pub trait InitChain<N: GenericNode>: ComponentInit<N> {
    fn init_and_return_root(self) -> N;
}

struct InitImpl<N, F>(PhantomData<N>, F);

impl<N, F> ComponentInit<N> for InitImpl<N, F>
where
    N: GenericNode,
    F: 'static + FnOnce() -> N,
{
    fn init(self) -> N {
        (self.1)()
    }
}

impl<N, F> InitChain<N> for InitImpl<N, F>
where
    N: GenericNode,
    F: 'static + FnOnce() -> N,
{
    fn init_and_return_root(self) -> N {
        (self.1)()
    }
}

pub trait RenderChain<N: GenericNode>: ComponentRender<N> {
    fn render_and_return_next(self, node: N) -> Option<N>;
}

struct RenderImpl<N, F>(PhantomData<N>, F);

impl<N, F> ComponentRender<N> for RenderImpl<N, F>
where
    N: GenericNode,
    F: 'static + FnOnce(N) -> Option<N>,
{
    fn render(self, node: N) {
        (self.1)(node);
    }
}

impl<N, F> RenderChain<N> for RenderImpl<N, F>
where
    N: GenericNode,
    F: 'static + FnOnce(N) -> Option<N>,
{
    fn render_and_return_next(self, node: N) -> Option<N> {
        (self.1)(node)
    }
}
