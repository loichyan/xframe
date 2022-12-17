use crate::{GenericNode, NodeType};
use ahash::AHashMap;
use std::{
    cell::{Cell, RefCell},
    fmt,
    rc::Rc,
};

thread_local! {
    static GLOBAL_ID: Cell<usize> = Cell::new(0);
}

pub struct Templates<N> {
    inner: Rc<RefCell<AHashMap<TemplateId, TemplateNode<N>>>>,
}

impl<N> Default for Templates<N> {
    fn default() -> Self {
        Self {
            inner: Default::default(),
        }
    }
}

impl<N> Clone for Templates<N> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TemplateId {
    id: usize,
}

impl Default for TemplateId {
    fn default() -> Self {
        Self::new()
    }
}

impl TemplateId {
    pub fn new() -> Self {
        Self {
            id: GLOBAL_ID.with(|id| {
                let current = id.get();
                id.set(current + 1);
                current
            }),
        }
    }
}

impl fmt::Display for TemplateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.id.fmt(f)
    }
}

#[derive(Clone)]
struct TemplateNode<N> {
    length: usize,
    container: N,
}

pub trait GenericComponent<N: GenericNode>: 'static + Sized {
    fn build_template(self) -> Template<N>;
    fn id() -> Option<TemplateId> {
        None
    }
    fn render(self) -> View<N> {
        self.into_dyn_component().render()
    }
    fn render_to(self, root: &N) {
        self.render().append_to(root);
    }
    fn into_dyn_component(self) -> DynComponent<N> {
        DynComponent {
            id: Self::id(),
            template: self.build_template(),
        }
    }
}

pub struct Template<N> {
    pub init: TemplateInit<N>,
    pub render: TemplateRender<N>,
}

pub struct TemplateInit<N>(Box<dyn FnOnce() -> View<N>>);

impl<N> TemplateInit<N> {
    pub fn new(f: impl 'static + FnOnce() -> View<N>) -> Self {
        Self(Box::new(f))
    }

    pub fn init(self) -> View<N> {
        (self.0)()
    }
}

pub struct TemplateRender<N>(Box<dyn FnOnce(Option<N>) -> Option<N>>);

impl<N> TemplateRender<N> {
    pub fn new(f: impl 'static + FnOnce(Option<N>) -> Option<N>) -> Self {
        Self(Box::new(f))
    }

    pub fn render(self, node: Option<N>) -> Option<N> {
        (self.0)(node)
    }
}

pub struct DynComponent<N> {
    id: Option<TemplateId>,
    template: Template<N>,
}

impl<N: GenericNode> DynComponent<N> {
    pub fn render(self) -> View<N> {
        let Self {
            id,
            template: Template { init, render },
        } = self;
        let TemplateNode { length, container } = {
            let create_template_node = move || {
                let container = N::create(NodeType::Template(id));
                let component = init.init();
                component.append_to(&container);
                TemplateNode {
                    length: component.len(),
                    container,
                }
            };
            if let Some(id) = id {
                // Initialize or reuse existing templates.
                let templates = N::global_templates();
                let mut templates = templates.inner.borrow_mut();
                let template = templates.entry(id).or_insert_with(create_template_node);
                TemplateNode {
                    length: template.length,
                    container: template.container.deep_clone(),
                }
            } else {
                // Initialize directly.
                create_template_node()
            }
        };
        if let Some(first) = container.first_child() {
            let last_child = render.render(Some(first.clone()));
            debug_assert!(last_child.is_none());
            if length == 1 {
                View::Node(first)
            } else {
                let mut fragment = Vec::with_capacity(length);
                let mut current = first.clone();
                fragment.push(first);
                while let Some(next) = current.next_sibling() {
                    fragment.push(next.clone());
                    current = next;
                }
                debug_assert_eq!(fragment.len(), length);
                View::Fragment(Rc::from(fragment.into_boxed_slice()))
            }
        } else {
            debug_assert_eq!(length, 0);
            View::empty()
        }
    }
}

impl<N: GenericNode> GenericComponent<N> for DynComponent<N> {
    fn build_template(self) -> Template<N> {
        self.template
    }

    fn into_dyn_component(self) -> DynComponent<N> {
        self
    }
}

#[derive(Clone)]
pub enum View<N> {
    Node(N),
    Fragment(Rc<[N]>),
}

impl<N: GenericNode> View<N> {
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
