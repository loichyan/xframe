use crate::node::GenericNode;
use std::rc::Rc;
use xframe_reactive::ReadSignal;

#[derive(Clone)]
pub struct View<N>(ViewType<N>);

#[derive(Clone)]
enum ViewType<N> {
    Node(N),
    Fragment(Rc<[View<N>]>),
    Dyn(ReadSignal<View<N>>),
}

use ViewType as VT;

impl<N: GenericNode> View<N> {
    pub fn node(node: N) -> Self {
        Self(VT::Node(node))
    }

    pub fn fragment(views: Vec<View<N>>) -> Self {
        Self(VT::Fragment(views.into_boxed_slice().into()))
    }

    pub fn fragment_shared(views: Rc<[View<N>]>) -> Self {
        if is_debug!() && views.is_empty() {
            panic!("empty `View` is not allowed")
        }
        Self(VT::Fragment(views))
    }

    pub(crate) fn dyn_(signal: ReadSignal<View<N>>) -> Self {
        Self(VT::Dyn(signal))
    }

    pub fn visit(&self, mut f: impl FnMut(&N)) {
        self.visit_impl(&mut f);
    }

    fn visit_impl(&self, f: &mut impl FnMut(&N)) {
        match &self.0 {
            VT::Node(t) => f(t),
            VT::Fragment(t) => t.iter().for_each(|t| t.visit_impl(f)),
            VT::Dyn(t) => t.get().visit_impl(f),
        }
    }

    pub fn deep_clone(&self) -> Self {
        match &self.0 {
            VT::Node(t) => Self(VT::Node(t.deep_clone())),
            VT::Fragment(t) => Self::fragment(t.iter().map(Self::deep_clone).collect()),
            VT::Dyn(t) => Self(VT::Dyn(*t)),
        }
    }

    pub fn ref_eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (VT::Node(t1), VT::Node(t2)) => t1.eq(t2),
            (VT::Fragment(t1), VT::Fragment(t2)) => Rc::ptr_eq(t1, t2),
            (VT::Dyn(t1), VT::Dyn(t2)) => t1.ref_eq(t2),
            _ => false,
        }
    }

    pub fn first(&self) -> N {
        let mut current = self.clone();
        loop {
            match current.0 {
                VT::Node(t) => return t,
                VT::Fragment(t) => current = t.first().unwrap().clone(),
                VT::Dyn(t) => current = t.get(),
            }
        }
    }

    pub fn last(&self) -> N {
        let mut current = self.clone();
        loop {
            match current.0 {
                VT::Node(t) => return t,
                VT::Fragment(t) => current = t.last().unwrap().clone(),
                VT::Dyn(t) => current = t.get(),
            }
        }
    }

    pub fn parent(&self) -> Option<N> {
        self.first().parent()
    }

    pub fn next_sibling(&self) -> Option<N> {
        self.last().next_sibling()
    }

    pub fn append_to(&self, parent: &N) {
        self.visit(|t| parent.append_child(t));
    }

    pub fn replace_with(&self, parent: &N, new_view: &Self) {
        debug_assert!(!self.ref_eq(new_view));
        if let (VT::Node(old), VT::Node(new)) = (&self.0, &new_view.0) {
            parent.replace_child(new, old);
        } else {
            new_view.move_before(parent, Some(self));
            self.remove_from(parent);
        }
    }

    pub fn remove_from(&self, parent: &N) {
        self.visit(|t| parent.remove_child(t));
    }

    pub fn move_before(&self, parent: &N, position: Option<&View<N>>) {
        debug_assert_ne!(position.map(|t| t.ref_eq(self)), Some(true));
        let position = position.map(View::first);
        self.visit(|t| parent.insert_before(t, position.as_ref()));
    }

    /// Visit all nodes in this view and check if they are mounted in the same order.
    pub fn check_mount_order(&self) -> bool {
        let mut same = true;
        let mut current = Some(self.first());
        self.visit(|node| {
            if node.parent().is_some() {
                if let Some(mounted) = current.as_ref() {
                    if mounted.ne(node) {
                        same = false;
                        current = None;
                    } else {
                        current = mounted.next_sibling();
                    }
                } else {
                    same = false;
                    current = None;
                }
            }
        });
        same
    }
}

/// Helper trait, node operations will be ignored when parent is [`None`].
pub trait ViewParentExt<N: GenericNode> {
    fn with_parent(&self, f: impl FnOnce(&N));

    fn append_child(&self, new_view: &View<N>) {
        self.with_parent(|parent| {
            new_view.remove_from(parent);
        });
    }

    fn replace_child(&self, new_view: &View<N>, old_view: &View<N>) {
        self.with_parent(|parent| {
            old_view.replace_with(parent, new_view);
        });
    }

    fn remove_child(&self, position: &View<N>) {
        self.with_parent(|parent| {
            position.remove_from(parent);
        });
    }

    fn insert_before(&self, new_view: &View<N>, position: Option<&View<N>>) {
        self.with_parent(|parent| {
            new_view.move_before(parent, position);
        });
    }
}

impl<N: GenericNode> ViewParentExt<N> for N {
    fn with_parent(&self, f: impl FnOnce(&N)) {
        f(self);
    }
}

impl<N: GenericNode> ViewParentExt<N> for Option<N> {
    fn with_parent(&self, f: impl FnOnce(&N)) {
        self.as_ref().with_parent(f);
    }
}

impl<N: GenericNode> ViewParentExt<N> for Option<&N> {
    fn with_parent(&self, f: impl FnOnce(&N)) {
        self.map(|n| n.with_parent(f));
    }
}
