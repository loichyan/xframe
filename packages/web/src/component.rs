use std::marker::PhantomData;
use xframe_core::{
    component::{ComponentInit, ComponentNode, ComponentRender, GenericComponent},
    Attribute, GenericElement, GenericNode, IntoReactive,
};
use xframe_reactive::Scope;

pub struct Component<Init, Render> {
    cx: Scope,
    init: Init,
    render: Render,
}

impl<N, Init, Render> Component<Init, Render>
where
    N: GenericNode,
    Init: InitChain<Node = N>,
    Render: RenderChain<Node = N>,
{
    pub fn child<C>(
        self,
        component: C,
    ) -> Component<impl InitChain<Node = N>, impl RenderChain<Node = N>>
    where
        C: GenericComponent<Node = N>,
    {
        let component = component.into_component_node();
        Component {
            cx: self.cx,
            init: InitImpl(PhantomData, move || {
                let root = self.init.init_and_return_root();
                root.append_child(component.init.init());
                root
            }),
            render: RenderImpl(PhantomData, move |prev: N| {
                let node = self.render.render_and_return_next(prev).unwrap();
                let next = node.next_sibling();
                component.render.render(node);
                next
            }),
        }
    }

    pub fn child_element<E, F>(
        self,
        render: F,
    ) -> Component<impl InitChain<Node = N>, impl RenderChain<Node = N>>
    where
        E: GenericElement<Node = N>,
        F: 'static + FnOnce(E),
    {
        let cx = self.cx;
        self.child(create_component(cx, render))
    }

    pub fn child_text<A: IntoReactive<Attribute>>(
        self,
        data: A,
    ) -> Component<impl InitChain<Node = N>, impl RenderChain<Node = N>> {
        let data = data.into_reactive();
        self.child_element(move |text: crate::elements::text<_>| {
            text.data(data);
        })
    }
}

impl<Init, Render> From<Component<Init, Render>> for ComponentNode<Init, Render> {
    fn from(t: Component<Init, Render>) -> Self {
        Self {
            init: t.init,
            render: t.render,
        }
    }
}

impl<N, Init, Render> GenericComponent for Component<Init, Render>
where
    N: GenericNode,
    Init: ComponentInit<Node = N>,
    Render: ComponentRender<Node = N>,
{
    type Node = N;
    type Init = Init;
    type Render = Render;
}

pub trait InitChain: 'static + ComponentInit {
    fn init_and_return_root(self) -> Self::Node;
}

struct InitImpl<N, F>(PhantomData<N>, F);

impl<N, F> ComponentInit for InitImpl<N, F>
where
    N: GenericNode,
    F: 'static + FnOnce() -> N,
{
    type Node = N;
    fn init(self) -> Self::Node {
        (self.1)()
    }
}

impl<N, F> InitChain for InitImpl<N, F>
where
    N: GenericNode,
    F: 'static + FnOnce() -> N,
{
    fn init_and_return_root(self) -> Self::Node {
        (self.1)()
    }
}

pub trait RenderChain: 'static + ComponentRender {
    fn render_and_return_next(self, node: Self::Node) -> Option<Self::Node>;
}

struct RenderImpl<N, F>(PhantomData<N>, F);

impl<N, F> ComponentRender for RenderImpl<N, F>
where
    N: GenericNode,
    F: 'static + FnOnce(N) -> Option<N>,
{
    type Node = N;
    fn render(self, node: Self::Node) {
        (self.1)(node);
    }
}

impl<N, F> RenderChain for RenderImpl<N, F>
where
    N: GenericNode,
    F: 'static + FnOnce(N) -> Option<N>,
{
    fn render_and_return_next(self, node: Self::Node) -> Option<Self::Node> {
        (self.1)(node)
    }
}

pub fn create_component<N, E>(
    cx: Scope,
    render: impl 'static + FnOnce(E),
) -> Component<impl InitChain<Node = N>, impl RenderChain<Node = N>>
where
    N: GenericNode,
    E: GenericElement<Node = N>,
{
    Component {
        cx,
        init: InitImpl(PhantomData, move || E::create(cx).into_node()),
        render: RenderImpl(PhantomData, move |root: N| {
            let next = root.first_child();
            render(E::create_with_node(cx, root));
            next
        }),
    }
}
