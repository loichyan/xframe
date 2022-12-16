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
        self.into_dyn_component().render()
    }

    fn into_dyn_component(self) -> DynComponent<N> {
        let ComponentNode { init, render } = self.into_component_node();
        DynComponent {
            id: TypeId::of::<Self::Identifier>(),
            init: Box::new(|| init.init()),
            render: Box::new(|node| render.render(node)),
        }
    }
}

type DynInit<N> = Box<dyn FnOnce() -> Component<N>>;
type DynRender<N> = Box<dyn FnOnce(Option<N>) -> Option<N>>;

pub struct DynComponent<N> {
    id: TypeId,
    init: DynInit<N>,
    render: DynRender<N>,
}

impl<N> From<DynComponent<N>> for ComponentNode<DynInit<N>, DynRender<N>> {
    fn from(t: DynComponent<N>) -> Self {
        ComponentNode {
            init: t.init,
            render: t.render,
        }
    }
}

impl<N: GenericNode> GenericComponent<N> for DynComponent<N> {
    type Init = DynInit<N>;
    type Render = DynRender<N>;
    type Identifier = ();

    fn render(self) -> Component<N> {
        let TemplateNode { length, container } = TEMPLATES.with(|templates| {
            templates
                .borrow_mut()
                .entry(self.id)
                .or_insert_with(|| {
                    let container = N::create(NodeType::Template);
                    let component = self.init.init();
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
            let last_child = self.render.render(Some(first.clone()));
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

    fn into_dyn_component(self) -> DynComponent<N> {
        self
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
    fn render(self, node: Option<N>) -> Option<N>;
}

impl<N, F> ComponentRender<N> for F
where
    N: GenericNode,
    F: 'static + FnOnce(Option<N>) -> Option<N>,
{
    fn render(self, node: Option<N>) -> Option<N> {
        (self)(node)
    }
}

#[derive(Clone)]
pub enum Component<N> {
    Node(N),
    Fragment(Rc<[N]>),
}

impl<N: GenericNode> Component<N> {
    pub fn empty() -> Self {
        Self::Fragment(Rc::new([]))
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Node(_) => 1,
            Self::Fragment(n) => n.len(),
        }
    }

    pub fn iter(&self) -> Iter<N> {
        match self {
            Self::Node(n) => Iter {
                inner: IterImpl::Node(Some(n)),
            },
            Self::Fragment(nodes) => Iter {
                inner: IterImpl::Fragment(nodes.iter()),
            },
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn first(&self) -> Option<&N> {
        self.iter().next()
    }

    pub fn insert_with(&mut self, f: impl FnOnce() -> N) {
        if self.is_empty() {
            *self = Self::Node(f())
        }
    }

    pub fn append_to(&self, parent: &N) {
        for node in self.iter() {
            parent.append_child(node);
        }
    }

    pub fn remove_from(&self, parent: &N) {
        for node in self.iter() {
            parent.remove_child(node);
        }
    }

    pub fn move_before(&self, parent: &N, ref_node: &N) {
        for node in self.iter() {
            parent.insert_before(node, ref_node);
        }
    }

    pub fn replace_with(&self, parent: &N, new_component: &Self) {
        match (self, new_component) {
            (Self::Node(old), Self::Node(new)) => parent.replace_child(new, old),
            _ => {
                if let Some(first) = self.first() {
                    new_component.move_before(parent, first);
                    self.remove_from(parent);
                }
            }
        }
    }
}

pub struct Iter<'a, N> {
    inner: IterImpl<'a, N>,
}

enum IterImpl<'a, N> {
    Node(Option<&'a N>),
    Fragment(std::slice::Iter<'a, N>),
}

impl<'a, N> Iterator for Iter<'a, N> {
    type Item = &'a N;
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            IterImpl::Node(n) => n.take(),
            IterImpl::Fragment(nodes) => nodes.next(),
        }
    }
}
