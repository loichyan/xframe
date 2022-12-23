use std::marker::PhantomData;
use xframe_core::{
    template::{BeforeRendering, RenderOutput, Template, TemplateInit, TemplateRender},
    Attribute, GenericComponent, GenericElement, GenericNode, IntoReactive, View,
};
use xframe_reactive::Scope;

pub fn view<N, E>(cx: Scope, _: fn(Scope) -> E) -> Element<N, E>
where
    N: GenericNode,
    E: GenericElement<N>,
{
    Element(cx)
}

#[allow(non_snake_case)]
pub fn Element<N, E>(cx: Scope) -> Element<N, E>
where
    N: GenericNode,
    E: GenericElement<N>,
{
    Element {
        cx,
        init: PhantomData,
        render: None,
        init_children: Box::new(|_| {}),
        render_children: Box::new(|first_child| first_child),
    }
}

pub struct Element<N, E> {
    cx: Scope,
    init: PhantomData<E>,
    render: Option<Box<dyn FnOnce(E) -> View<N>>>,
    init_children: Box<dyn FnOnce(&N)>,
    render_children: Box<dyn FnOnce(Option<N>) -> Option<N>>,
}

impl<N, E> GenericComponent<N> for Element<N, E>
where
    N: GenericNode,
    E: GenericElement<N>,
{
    fn build_template(self) -> Template<N> {
        let Self {
            cx,
            render,
            init_children,
            render_children,
            ..
        } = self;
        Template {
            init: TemplateInit::<N>::new(move || {
                let root = E::create(cx);
                let root = root.into_node();
                init_children(&root);
                View::node(root)
            }),
            render: TemplateRender::<N>::new(move |before_rendering, root| {
                let first_child = root.first_child();
                let output = RenderOutput {
                    // Save the next sibling be rendering.
                    next: root.next_sibling(),
                    view: {
                        before_rendering.apply_to(&root);
                        if let Some(render) = render {
                            render(E::create_with_node(cx, root))
                        } else {
                            View::node(root)
                        }
                    },
                };
                let last_child = render_children(first_child);
                // All children should be visited.
                debug_assert!(last_child.is_none());
                output
            }),
        }
    }
}

impl<N, E> Element<N, E>
where
    N: GenericNode,
    E: GenericElement<N>,
{
    pub fn build(self) -> Self {
        self
    }

    pub fn with(self, render: impl 'static + FnOnce(E) -> E) -> Self {
        self.with_view(move |el| View::node(render(el).into_node()))
    }

    pub fn with_view(mut self, render: impl 'static + FnOnce(E) -> View<N>) -> Self {
        if self.render.is_some() {
            panic!("`Element::with_view` has already been specified");
        }
        self.render = Some(Box::new(render));
        self
    }

    pub fn child<C: GenericComponent<N>>(mut self, child: C) -> Self {
        let Template { init, render, .. } = child.build_template();
        self.init_children = Box::new(move |root| {
            (self.init_children)(root);
            init.init().append_to(root);
        });
        self.render_children = Box::new(move |first| {
            let node = (self.render_children)(first);
            render.render(BeforeRendering::Nothing, node.unwrap()).next
        });
        self
    }

    pub fn child_element<Child>(self, render: impl 'static + FnOnce(Child) -> Child) -> Self
    where
        Child: GenericElement<N>,
    {
        let cx = self.cx;
        self.child(Element(cx).with(render))
    }

    pub fn child_text<A: IntoReactive<Attribute>>(self, data: A) -> Self {
        let data = data.into_reactive();
        self.child_element(move |text: crate::elements::text<_>| text.data(data))
    }
}
