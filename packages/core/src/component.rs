use crate::{node::NodeType, GenericNode};
use ahash::AHashMap;
use std::{
    any::{Any, TypeId},
    cell::RefCell,
};

type Templates = RefCell<AHashMap<TypeId, Box<dyn Any>>>;

thread_local! {
    static TEMPLATES: Templates = Templates::default();
}

pub struct ComponentNode<Init, Render> {
    pub init: Init,
    pub render: Render,
}

pub trait GenericComponent<N: GenericNode>:
    'static + Into<ComponentNode<Self::Init, Self::Render>>
{
    type Init: ComponentInit<N>;
    type Render: ComponentRender<N>;
    type Identifier: 'static;

    fn into_component_node(self) -> ComponentNode<Self::Init, Self::Render> {
        self.into()
    }

    fn render(self) -> N {
        let component = self.into_component_node();
        let node = TEMPLATES.with(|templates| {
            templates
                .borrow_mut()
                .entry(TypeId::of::<Self::Identifier>())
                .or_insert_with(|| {
                    let fragment = N::create(NodeType::Fragment);
                    fragment.append_child(&component.init.init());
                    Box::new(fragment)
                })
                .downcast_ref::<N>()
                .unwrap_or_else(|| unreachable!())
                .first_child()
                .unwrap_or_else(|| unreachable!())
                .deep_clone()
        });
        component.render.render(node.clone());
        node
    }
}

pub trait ComponentInit<N: GenericNode>: 'static {
    fn init(self) -> N;
}

pub trait ComponentRender<N: GenericNode>: 'static {
    fn render(self, node: N);
}
