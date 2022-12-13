use crate::{element::GenericElement, node::GenericNode};
use ahash::AHashMap;
use std::{
    any::{Any, TypeId},
    cell::RefCell,
    marker::PhantomData,
};
use xframe_reactive::Scope;

type Templates = RefCell<AHashMap<TypeId, Box<dyn Any>>>;

thread_local! {
    static TEMPLATES: Templates = Templates::default();
}

pub struct Component<Init, Render> {
    cx: Scope,
    init: Init,
    render: Render,
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
        init: InitChainImpl(PhantomData, move || E::create(cx).into_node()),
        render: RenderChainImpl(PhantomData, move |node: N| {
            let next = node.first_child();
            render(E::create_with_node(cx, node));
            next
        }),
    }
}

impl<N, Init, Render> Component<Init, Render>
where
    N: GenericNode,
    Init: InitChain<Node = N>,
    Render: RenderChain<Node = N>,
{
    pub fn child(
        self,
        component: impl IntoComponent<Node = N>,
    ) -> Component<impl InitChain<Node = N>, impl RenderChain<Node = N>> {
        let child = component.into_component();
        Component {
            cx: self.cx,
            init: InitChainImpl(PhantomData, move || {
                let node = self.init.init();
                node.append_child(child.init.init());
                node
            }),
            render: RenderChainImpl(PhantomData, move |mut node: N| {
                node = self.render.render(node).unwrap();
                let next = node.next_sibling();
                child.render.render(node);
                next
            }),
        }
    }
}

pub trait IntoComponent: 'static + Sized {
    type Node: GenericNode;
    type Init: InitChain<Node = Self::Node>;
    type Render: RenderChain<Node = Self::Node>;

    fn into_component(self) -> Component<Self::Init, Self::Render>;

    fn render(self) -> Self::Node {
        let component = self.into_component();
        let node = TEMPLATES.with(|templates| {
            // Reuse existed templates or create a new one.
            templates
                .borrow_mut()
                .entry(TypeId::of::<Self>())
                .or_insert_with(move || {
                    let fragment = Self::Node::create_fragment();
                    fragment.append_child(component.init.init());
                    Box::new(fragment)
                })
                .downcast_ref::<Self::Node>()
                .unwrap_or_else(|| unreachable!())
                .first_child()
                .unwrap_or_else(|| unreachable!())
                .deep_clone()
        });
        component.render.render(node.clone());
        node
    }
}

impl<N, Init, Render> IntoComponent for Component<Init, Render>
where
    N: GenericNode,
    Init: InitChain<Node = N>,
    Render: RenderChain<Node = N>,
{
    type Node = N;
    type Init = Init;
    type Render = Render;

    fn into_component(self) -> Component<Self::Init, Self::Render> {
        self
    }
}

pub trait InitChain: 'static {
    type Node: GenericNode;

    fn init(self) -> Self::Node;
}

struct InitChainImpl<N, F>(PhantomData<N>, F);

impl<N, F> InitChain for InitChainImpl<N, F>
where
    N: GenericNode,
    F: 'static + FnOnce() -> N,
{
    type Node = N;
    fn init(self) -> Self::Node {
        (self.1)()
    }
}

pub trait RenderChain: 'static {
    type Node: GenericNode;

    fn render(self, node: Self::Node) -> Option<Self::Node>;
}

struct RenderChainImpl<N, F>(PhantomData<N>, F);

impl<N, F> RenderChain for RenderChainImpl<N, F>
where
    N: GenericNode,
    F: 'static + FnOnce(N) -> Option<N>,
{
    type Node = N;

    fn render(self, node: N) -> Option<N> {
        (self.1)(node)
    }
}
