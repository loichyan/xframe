use crate::{node::NodeType, GenericNode};
use ahash::AHashMap;
use std::{
    any::{Any, TypeId},
    cell::RefCell,
    rc::Rc,
};

type Templates = RefCell<AHashMap<TypeId, Box<dyn Any>>>;

#[derive(Clone)]
struct TemplateNode<N> {
    length: usize,
    container: N,
}

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

    fn render(self) -> Component<N> {
        let component = self.into_component_node();
        let TemplateNode { length, container } = TEMPLATES.with(|templates| {
            templates
                .borrow_mut()
                .entry(TypeId::of::<Self::Identifier>())
                .or_insert_with(|| {
                    let container = N::create(NodeType::Fragment);
                    let component = component.init.init();
                    component.append_to(&container);
                    Box::new(TemplateNode {
                        length: component.len(),
                        container,
                    })
                })
                .downcast_ref::<TemplateNode<N>>()
                .map(|tmpl| TemplateNode {
                    length: tmpl.length,
                    container: tmpl.container.deep_clone(),
                })
                .unwrap_or_else(|| unreachable!())
        });
        if let Some(first) = container.first_child() {
            let last_child = component.render.render(first.clone());
            debug_assert!(last_child.is_none());
            if length == 1 {
                Component::Node(first)
            } else {
                let mut fragment = Vec::with_capacity(length);
                let mut current = first.clone();
                fragment.push(first);
                while let Some(next) = current.next_sibling() {
                    fragment.push(next.clone());
                    current = next;
                }
                debug_assert_eq!(fragment.len(), length);
                Component::Fragment(Rc::from(fragment.into_boxed_slice()))
            }
        } else {
            debug_assert_eq!(length, 0);
            Component::Fragment(Rc::new([]))
        }
    }
}

pub trait ComponentInit<N: GenericNode>: 'static {
    fn init(self) -> Component<N>;
}

impl<N, F> ComponentInit<N> for F
where
    N: GenericNode,
    F: 'static + FnOnce() -> Component<N>,
{
    fn init(self) -> Component<N> {
        (self)()
    }
}

pub trait ComponentRender<N: GenericNode>: 'static {
    /// Render and return **the next sibling**.
    fn render(self, node: N) -> Option<N>;
}

impl<N, F> ComponentRender<N> for F
where
    N: GenericNode,
    F: 'static + FnOnce(N) -> Option<N>,
{
    fn render(self, node: N) -> Option<N> {
        (self)(node)
    }
}

#[derive(Clone)]
pub enum Component<N> {
    Node(N),
    Fragment(Rc<[N]>),
}

impl<N: GenericNode> Component<N> {
    pub fn len(&self) -> usize {
        match self {
            Self::Node(_) => 1,
            Self::Fragment(n) => n.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn first(&self) -> Option<&N> {
        match self {
            Self::Node(n) => Some(n),
            Self::Fragment(nodes) => nodes.first(),
        }
    }

    pub fn append_to(&self, parent: &N) {
        match self {
            Self::Node(n) => parent.append_child(n),
            Self::Fragment(nodes) => {
                for n in nodes.iter() {
                    parent.append_child(n);
                }
            }
        }
    }

    pub fn replace_with(&self, parent: &N, new: &Self) {
        match (self, new) {
            (Self::Node(old), Self::Node(new)) => parent.replace_child(new, old),
            _ => todo!(),
        }
    }
}
