use crate::{
    node::{GenericNode, NodeType},
    template::*,
    view::View,
};

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
        let TemplateNode { container, .. } = {
            let create_template_node = move || {
                let container = N::create(NodeType::Template(id));
                let view = init.init();
                view.append_to(&container);
                TemplateNode { view, container }
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
            let TemplateRenderOutput { next_sibling, view } = render.render(Some(first));
            debug_assert!(next_sibling.is_none());
            view
        } else {
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
