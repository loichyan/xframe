use crate::node::GenericNode;

pub struct ComponentNode<Init, Render> {
    pub init: Init,
    pub render: Render,
}

pub trait GenericComponent: 'static + Into<ComponentNode<Self::Init, Self::Render>> {
    type Node: GenericNode;
    type Init: ComponentInit<Node = Self::Node>;
    type Render: ComponentRender<Node = Self::Node>;

    fn render(self) -> Self::Node;
    fn into_component_node(self) -> ComponentNode<Self::Init, Self::Render> {
        self.into()
    }
}

pub trait ComponentInit: 'static {
    type Node: GenericNode;
    fn init_and_return_root(self) -> Self::Node;
}

pub trait ComponentRender: 'static {
    type Node: GenericNode;
    fn render_and_return_next(self, node: Self::Node) -> Option<Self::Node>;
}
