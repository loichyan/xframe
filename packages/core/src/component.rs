use crate::GenericNode;
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

pub trait GenericComponent: 'static + Into<ComponentNode<Self::Init, Self::Render>> {
    type Node: GenericNode;
    type Init: ComponentInit<Node = Self::Node>;
    type Render: ComponentRender<Node = Self::Node>;
    type Identifier: 'static;

    fn into_component_node(self) -> ComponentNode<Self::Init, Self::Render> {
        self.into()
    }

    fn render(self) -> Self::Node {
        let component = self.into_component_node();
        let node = TEMPLATES.with(|templates| {
            templates
                .borrow_mut()
                .entry(TypeId::of::<Self::Identifier>())
                .or_insert_with(|| {
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

pub trait ComponentInit: 'static {
    type Node: GenericNode;
    fn init(self) -> Self::Node;
}

pub trait ComponentRender: 'static {
    type Node: GenericNode;
    fn render(self, node: Self::Node);
}
