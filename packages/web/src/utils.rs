mod visit;

use std::{cell::RefCell, rc::Rc};

use self::visit::VisitSkip;

macro_rules! define_placeholder {
    ($vis:vis $name:ident($desc:literal)) => {
        $vis struct $name<N> {
            node: N,
        }

        #[allow(dead_code)]
        impl<N: ::xframe_core::GenericNode> $name<N> {
            pub fn new() -> Self {
                Self {
                    node: N::create(<Self as ::xframe_core::GenericElement<N>>::TYPE),
                }
            }
        }

        impl<N: ::xframe_core::GenericNode> xframe_core::GenericElement<N> for $name<N> {
            const TYPE: ::xframe_core::NodeType =
                ::xframe_core::NodeType::Placeholder(::std::borrow::Cow::Borrowed($desc));

            fn create_with_node(_: ::xframe_reactive::Scope, node: N) -> Self {
                Self { node }
            }

            fn into_node(self) -> N {
                self.node
            }
        }
    };
}

pub trait Visit<T> {
    fn visit(&self, f: impl FnMut(&T));

    fn count(&self) -> usize {
        let mut i = 0;
        self.visit(|_| i += 1);
        i
    }

    fn skip(&self, count: usize) -> VisitSkip<Self> {
        VisitSkip {
            visitor: self,
            count,
        }
    }
}

impl<T, U: Visit<T>> Visit<T> for Rc<U> {
    fn visit(&self, f: impl FnMut(&T)) {
        U::visit(self, f);
    }

    fn count(&self) -> usize {
        U::count(self)
    }
}

impl<T, U: Visit<T>> Visit<T> for RefCell<U> {
    fn visit(&self, f: impl FnMut(&T)) {
        self.borrow().visit(f);
    }

    fn count(&self) -> usize {
        U::count(&self.borrow())
    }
}

impl<T> Visit<T> for Vec<T> {
    fn visit(&self, f: impl FnMut(&T)) {
        self.iter().for_each(f);
    }

    fn count(&self) -> usize {
        self.len()
    }
}
