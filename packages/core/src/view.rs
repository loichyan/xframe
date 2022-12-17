use crate::GenericNode;
use std::rc::Rc;

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
