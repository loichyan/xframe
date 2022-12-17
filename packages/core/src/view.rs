use xframe_reactive::Signal;

use crate::node::GenericNode;
use std::rc::Rc;

#[derive(Clone)]
pub enum View<N> {
    Node(N),
    Fragment(Rc<[View<N>]>),
    Dyn(Rc<dyn Fn() -> View<N>>),
}

impl<N: GenericNode> From<Signal<View<N>>> for View<N> {
    fn from(t: Signal<View<N>>) -> Self {
        View::from(move || t.get())
    }
}

impl<N, F> From<F> for View<N>
where
    F: 'static + Fn() -> View<N>,
{
    fn from(f: F) -> Self {
        View::Dyn(Rc::new(f))
    }
}

impl<N: GenericNode> View<N> {
    pub fn empty() -> Self {
        Self::Fragment(Rc::new([]))
    }

    pub fn len(&self) -> usize {
        let mut i = 0;
        self.visit(|_| i += 1);
        i
    }

    pub fn visit(&self, mut f: impl FnMut(&N)) {
        self.visit_impl(&mut f);
    }

    fn visit_impl(&self, f: &mut impl FnMut(&N)) {
        match self {
            Self::Node(v) => f(v),
            Self::Fragment(v) => v.iter().for_each(|v| v.visit_impl(f)),
            Self::Dyn(v) => v().visit_impl(f),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn first(&self) -> N {
        match self {
            Self::Node(v) => v.clone(),
            Self::Fragment(v) => v
                .first()
                .unwrap_or_else(|| panic!("`View` cannot be empty"))
                .first(),
            Self::Dyn(v) => v().first(),
        }
    }

    fn ensure_not_empty(&self) {
        if cfg!(debug_assertions) && self.is_empty() {
            panic!("`View` cannot be empty")
        }
    }

    pub fn append_to(&self, parent: &N) {
        self.ensure_not_empty();
        self.visit(|node| parent.append_child(node));
    }

    pub fn remove_from(&self, parent: &N) {
        self.ensure_not_empty();
        self.visit(|node| parent.remove_child(node));
    }

    pub fn move_before(&self, parent: &N, ref_node: &N) {
        self.ensure_not_empty();
        self.visit(|node| parent.insert_before(node, ref_node));
    }

    pub fn replace_with(&self, parent: &N, new_component: &Self) {
        match (self, new_component) {
            (Self::Node(old), Self::Node(new)) => parent.replace_child(new, old),
            _ => {
                new_component.move_before(parent, &self.first());
                self.remove_from(parent);
            }
        }
    }
}
