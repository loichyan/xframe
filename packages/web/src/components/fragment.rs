use crate::Element;
use xframe_core::{
    template::{BeforeRendering, RenderOutput, Template, TemplateInit, TemplateRender},
    Attribute, GenericComponent, GenericElement, GenericNode, IntoReactive, View,
};
use xframe_reactive::Scope;

define_placeholder!(Placeholder("PLACEHOLDER FOR `xframe::Fragment` COMPONENT"));

type Views<N> = Vec<View<N>>;

#[allow(non_snake_case)]
pub fn Fragment<N: GenericNode>(cx: Scope) -> Fragment<N> {
    Fragment {
        cx,
        init: None,
        render: Box::new(|_, first_node, _| Some(first_node)),
    }
}

pub struct Fragment<N> {
    cx: Scope,
    init: Option<Box<dyn FnOnce(&N)>>,
    render: Box<dyn FnOnce(BeforeRendering<N>, N, &mut Views<N>) -> Option<N>>,
}

impl<N: GenericNode> GenericComponent<N> for Fragment<N> {
    fn build_template(self) -> Template<N> {
        let Self { init, render, .. } = self;
        Template {
            init: TemplateInit::new(move |parent| {
                if let Some(init) = init {
                    init(parent);
                } else {
                    // Fallback to a placeholder.
                    parent.append_child(&N::create(Placeholder::<N>::TYPE));
                }
            }),
            render: TemplateRender::new(move |before_rendering, node| {
                let mut views = Views::default();
                let next = render(before_rendering, node, &mut views);
                if views.is_empty() {
                    // Ignore the placeholder.
                    let placeholder = next.unwrap();
                    RenderOutput {
                        next: placeholder.next_sibling(),
                        view: {
                            before_rendering.apply_to(&placeholder);
                            View::node(placeholder)
                        },
                    }
                } else {
                    RenderOutput {
                        next,
                        view: View::fragment(views),
                    }
                }
            }),
        }
    }
}

impl<N: GenericNode> Fragment<N> {
    pub fn build(self) -> Self {
        self
    }

    pub fn child<C: GenericComponent<N>>(mut self, child: C) -> Self {
        let Template { init, render, .. } = child.build_template();
        self.init = Some(Box::new(move |parent| {
            if let Some(init) = self.init {
                init(parent);
            }
            init.init(parent);
        }));
        self.render = Box::new(move |before_rendering, first, views| {
            let node = (self.render)(before_rendering, first, views);
            let RenderOutput { next, view } = render.render(before_rendering, node.unwrap());
            views.push(view);
            next
        });
        self
    }

    pub fn child_element<E>(self, render: impl 'static + FnOnce(E) -> E) -> Self
    where
        E: GenericElement<N>,
    {
        let cx = self.cx;
        self.child(Element(cx).with(render))
    }

    pub fn child_text<A: IntoReactive<Attribute>>(self, data: A) -> Self {
        let data = data.into_reactive(self.cx);
        self.child_element(move |text: crate::elements::text<_>| text.data(data))
    }
}
