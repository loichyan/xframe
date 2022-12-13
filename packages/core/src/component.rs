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

pub struct ComponentNode<Init, Render> {
    init: Init,
    render: Render,
}

pub trait Component: 'static + Into<ComponentNode<Self::Init, Self::Render>> {
    type Node: GenericNode;
    type Init: InitChain<Node = Self::Node>;
    type Render: RenderChain<Node = Self::Node>;
    type ChildOutput<C: Component<Node = Self::Node>>: Component<Node = Self::Node>;
    type ChildElementOutput<E: GenericElement<Node = Self::Node>, F: 'static + FnOnce(E)>: Component<
        Node = Self::Node,
    >;

    fn render(self) -> Self::Node;
    fn child<C>(self, component: C) -> Self::ChildOutput<C>
    where
        C: Component<Node = Self::Node>;
    fn child_element<E, F>(self, render: F) -> Self::ChildElementOutput<E, F>
    where
        E: GenericElement<Node = Self::Node>,
        F: 'static + FnOnce(E);
}

struct ComponentImpl<Init, Render> {
    cx: Scope,
    init: Init,
    render: Render,
}

impl<Init, Render> From<ComponentImpl<Init, Render>> for ComponentNode<Init, Render> {
    fn from(t: ComponentImpl<Init, Render>) -> Self {
        Self {
            init: t.init,
            render: t.render,
        }
    }
}

impl<N, Init, Render> Component for ComponentImpl<Init, Render>
where
    N: GenericNode,
    Init: InitChain<Node = N>,
    Render: RenderChain<Node = N>,
{
    type Node = N;
    type Init = Init;
    type Render = Render;
    type ChildElementOutput<E: GenericElement<Node = Self::Node>, F: 'static + FnOnce(E)> =
        ComponentImpl<InitChain2<Init, InitElement<E>>, RenderChain2<Render, RenderElement<E, F>>>;
    type ChildOutput<C: Component<Node = Self::Node>> =
        ComponentImpl<InitChain2<Init, C::Init>, RenderChain2<Render, C::Render>>;

    fn render(self) -> Self::Node {
        let node = TEMPLATES.with(|templates| {
            templates
                .borrow_mut()
                .entry(TypeId::of::<Self>())
                .or_insert_with(|| {
                    let fragment = Self::Node::create_fragment();
                    fragment.append_child(self.init.append_and_return_root(self.cx));
                    Box::new(fragment)
                })
                .downcast_ref::<Self::Node>()
                .unwrap_or_else(|| unreachable!())
                .first_child()
                .unwrap_or_else(|| unreachable!())
                .deep_clone()
        });
        self.render.render_and_return_next(self.cx, node.clone());
        node
    }

    fn child<C>(self, component: C) -> Self::ChildOutput<C>
    where
        C: Component<Node = Self::Node>,
    {
        let component: ComponentNode<_, _> = component.into();
        ComponentImpl {
            cx: self.cx,
            init: InitChain2(self.init, component.init),
            render: RenderChain2(self.render, component.render),
        }
    }

    fn child_element<E, F>(self, render: F) -> Self::ChildElementOutput<E, F>
    where
        E: GenericElement<Node = Self::Node>,
        F: 'static + FnOnce(E),
    {
        ComponentImpl {
            cx: self.cx,
            init: InitChain2(self.init, InitElement(PhantomData)),
            render: RenderChain2(self.render, RenderElement(PhantomData, render)),
        }
    }
}

pub trait InitChain: 'static {
    type Node: GenericNode;
    fn append_and_return_root(self, cx: Scope) -> Self::Node;
}

struct InitElement<E>(PhantomData<E>);
impl<N, E> InitChain for InitElement<E>
where
    N: GenericNode,
    E: GenericElement<Node = N>,
{
    type Node = N;
    fn append_and_return_root(self, cx: Scope) -> Self::Node {
        E::create(cx).into_node()
    }
}

struct InitChain2<Prev, This>(Prev, This);
impl<N, Prev, This> InitChain for InitChain2<Prev, This>
where
    N: GenericNode,
    Prev: InitChain<Node = N>,
    This: InitChain<Node = N>,
{
    type Node = N;
    fn append_and_return_root(self, cx: Scope) -> Self::Node {
        // Append previous siblings first.
        let root = self.0.append_and_return_root(cx);
        root.append_child(self.1.append_and_return_root(cx));
        root
    }
}

pub trait RenderChain: 'static {
    type Node: GenericNode;
    fn render_and_return_next(self, cx: Scope, prev: Self::Node) -> Option<Self::Node>;
}

struct RenderElement<E, F>(PhantomData<E>, F);
impl<N, E, F> RenderChain for RenderElement<E, F>
where
    N: GenericNode,
    E: GenericElement<Node = N>,
    F: 'static + FnOnce(E),
{
    type Node = N;
    fn render_and_return_next(self, cx: Scope, prev: Self::Node) -> Option<Self::Node> {
        let next = prev.first_child();
        (self.1)(E::create_with_node(cx, prev));
        next
    }
}

struct RenderChain2<Prev, This>(Prev, This);
impl<N, Prev, This> RenderChain for RenderChain2<Prev, This>
where
    N: GenericNode,
    Prev: RenderChain<Node = N>,
    This: RenderChain<Node = N>,
{
    type Node = N;
    fn render_and_return_next(self, cx: Scope, prev: N) -> Option<N> {
        // Render previous siblings and get this node.
        let node = self.0.render_and_return_next(cx, prev).unwrap();
        let next = node.next_sibling();
        self.1.render_and_return_next(cx, node);
        next
    }
}

pub fn create_component<N, E>(
    cx: Scope,
    render: impl 'static + FnOnce(E),
) -> impl Component<Node = N>
where
    N: GenericNode,
    E: GenericElement<Node = N>,
{
    ComponentImpl {
        cx,
        init: InitElement::<E>(PhantomData),
        render: RenderElement::<E, _>(PhantomData, render),
    }
}
