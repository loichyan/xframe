use xframe_reactive::untrack;

use crate::{
    is_dev,
    node::{GenericNode, NodeType},
    template::*,
    view::View,
};

pub trait GenericComponent<N: GenericNode>: 'static + Sized {
    fn build_template(self) -> Template<N>;

    fn id() -> Option<TemplateId> {
        None
    }

    fn into_dyn_component(self) -> DynComponent<N> {
        DynComponent {
            id: Self::id(),
            template: self.build_template(),
        }
    }

    /// Initialize and render the template, all nodes in returned [`View`] don't
    /// have parent nodes.
    fn render(self) -> View<N> {
        self.into_dyn_component().render()
    }

    /// Initialize the template and mount all nodes to `parent`, then perform rendering.
    fn mount_to(self, parent: &N) -> View<N> {
        self.into_dyn_component().mount_to(parent)
    }
}

/// Earse types and pack [`GenericComponent`] into a single type.
///
/// NOTE: you should call the [`DynComponent::id`] instead of [`GenericComponent::id`]
/// to get the [`TemplateId`] of the source component.
pub struct DynComponent<N: GenericNode> {
    id: Option<TemplateId>,
    template: Template<N>,
}

impl<N: GenericNode> GenericComponent<N> for DynComponent<N> {
    fn id() -> Option<TemplateId> {
        if is_dev!() {
            panic!("`<DynComponent as GenericComponent>::id` should not be invoked, use `DynComponent::id` instead")
        }
        None
    }

    fn build_template(self) -> Template<N> {
        self.template
    }

    fn into_dyn_component(self) -> DynComponent<N> {
        self
    }

    fn render(self) -> View<N> {
        self.render_impl(None)
    }

    fn mount_to(self, parent: &N) -> View<N> {
        self.render_impl(Some(parent))
    }
}

impl<N: GenericNode> DynComponent<N> {
    pub fn id(&self) -> Option<TemplateId> {
        self.id
    }

    fn render_impl(self, parent: Option<&N>) -> View<N> {
        let Self {
            id,
            template: Template { init, render },
        } = self;
        // 1) Initialize template.
        let TemplateContent { container } = {
            let init_template = move || {
                let container = N::create(NodeType::Template(
                    id.map(|id| id.data()).unwrap_or("").into(),
                ));
                init.init().append_to(&container);
                TemplateContent { container }
            };
            if let Some(id) = id {
                // Get a copy from the global templates.
                GlobalTemplates::clone_or_insert_with(id, init_template)
            } else {
                // Initialize the template without caching.
                init_template()
            }
        };

        // 2) Render the template.
        let before_rendering = parent
            // If parent is provided, all nodes of the component should be appended to it.
            .map(BeforeRendering::AppendTo)
            // Otherwise, they should be removed from the container.
            .unwrap_or(BeforeRendering::RemoveFrom(&container));
        let RenderOutput { view, next } = {
            // Ignore the side effect while accessing dynamic views.
            untrack(|| render.render(before_rendering, container.first_child().unwrap()))
        };
        // All child nodes in the container should be moved or deleted.
        debug_assert!(container.first_child().is_none());
        // All child nodes should be visited.
        debug_assert!(next.is_none());
        view
    }
}
