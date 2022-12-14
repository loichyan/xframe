use ahash::AHashMap;
use std::{
    any::{Any, TypeId},
    cell::RefCell,
    marker::PhantomData,
};
use xframe_core::{
    component::{ComponentInit, ComponentNode, ComponentRender, GenericComponent},
    Attribute, GenericElement, GenericNode, IntoReactive,
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

impl<N, Init, Render> Component<Init, Render>
where
    N: GenericNode,
    Init: ComponentInit<Node = N>,
    Render: ComponentRender<Node = N>,
{
    pub fn child<C>(
        self,
        component: C,
    ) -> Component<impl ComponentInit<Node = N>, impl ComponentRender<Node = N>>
    where
        C: GenericComponent<Node = N>,
    {
        let component = component.into_component_node();
        Component {
            cx: self.cx,
            init: InitImpl(PhantomData, move || {
                let root = self.init.init_and_return_root();
                root.append_child(component.init.init_and_return_root());
                root
            }),
            render: RenderImpl(PhantomData, move |prev: N| {
                let node = self.render.render_and_return_next(prev).unwrap();
                let next = node.next_sibling();
                let child_next = component.render.render_and_return_next(node);
                debug_assert!(child_next.is_none());
                next
            }),
        }
    }

    pub fn child_element<E, F>(
        self,
        render: F,
    ) -> Component<impl ComponentInit<Node = N>, impl ComponentRender<Node = N>>
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
    ) -> Component<impl ComponentInit<Node = N>, impl ComponentRender<Node = N>> {
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

    fn render(self) -> Self::Node {
        let node = TEMPLATES.with(|templates| {
            templates
                .borrow_mut()
                .entry(TypeId::of::<Self>())
                .or_insert_with(|| {
                    let fragment = Self::Node::create_fragment();
                    fragment.append_child(self.init.init_and_return_root());
                    Box::new(fragment)
                })
                .downcast_ref::<Self::Node>()
                .unwrap_or_else(|| unreachable!())
                .first_child()
                .unwrap_or_else(|| unreachable!())
                .deep_clone()
        });
        let child_next = self.render.render_and_return_next(node.clone());
        debug_assert!(child_next.is_none());
        node
    }
}

struct InitImpl<N, F>(PhantomData<N>, F);
impl<N, F> ComponentInit for InitImpl<N, F>
where
    N: GenericNode,
    F: 'static + FnOnce() -> N,
{
    type Node = N;
    fn init_and_return_root(self) -> Self::Node {
        (self.1)()
    }
}

struct RenderImpl<N, F>(PhantomData<N>, F);
impl<N, F> ComponentRender for RenderImpl<N, F>
where
    N: GenericNode,
    F: 'static + FnOnce(N) -> Option<N>,
{
    type Node = N;
    fn render_and_return_next(self, node: Self::Node) -> Option<N> {
        (self.1)(node)
    }
}

pub fn create_component<N, E>(
    cx: Scope,
    render: impl 'static + FnOnce(E),
) -> Component<impl ComponentInit<Node = N>, impl ComponentRender<Node = N>>
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
