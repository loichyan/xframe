use crate::{
    node::{GenericNode, NodeType},
    template::*,
    view::View,
};
use std::rc::Rc;

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
                N::global_templates().get_or_insert_with(id, create_template_node)
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
