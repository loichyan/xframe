use xframe_core::{
    template::{Template, TemplateInit, TemplateRender, TemplateRenderOutput},
    Attribute, GenericComponent, GenericElement, GenericNode, IntoReactive, View,
};
use xframe_reactive::Scope;

pub fn view<N, E>(cx: Scope, render: impl 'static + FnOnce(E) -> E) -> Element<N>
where
    N: GenericNode,
    E: GenericElement<N>,
{
    Element(cx).root(|root| View::Node(render(root).into_node()))
}

pub fn view_with<N, E>(cx: Scope, render: impl 'static + FnOnce(E) -> View<N>) -> Element<N>
where
    N: GenericNode,
    E: GenericElement<N>,
{
    Element(cx).root(render)
}

#[allow(non_snake_case)]
pub fn Element<N: GenericNode>(cx: Scope) -> Element<N> {
    Element {
        cx,
        init: ElementInit::new(|| panic!("The root element is not specified!")),
        render: ElementRender::new(|_| panic!("The root element is not specified!")),
    }
}

pub struct Element<N> {
    cx: Scope,
    init: ElementInit<N>,
    render: ElementRender<N>,
}

impl<N: GenericNode> Element<N> {
    pub fn root<E>(self, render: impl 'static + FnOnce(E) -> View<N>) -> Self
    where
        E: GenericElement<N>,
    {
        let cx = self.cx;
        Self {
            cx,
            init: ElementInit::new(|| N::create(E::TYPE)),
            render: ElementRender::<N>::new(move |root| {
                let root = root.unwrap_or_else(|| unreachable!());
                (
                    root.first_child(),
                    TemplateRenderOutput {
                        next_sibling: root.next_sibling(),
                        view: render(E::create_with_node(cx, root)),
                    },
                )
            }),
        }
    }

    pub fn build(self) -> impl GenericComponent<N> {
        self
    }

    pub fn child<C>(self, child: C) -> Element<N>
    where
        C: GenericComponent<N>,
    {
        let child = child.build_template();
        Element {
            cx: self.cx,
            init: self.init.child(child.init),
            render: self.render.child(child.render),
        }
    }

    pub fn child_element<E>(self, render: impl 'static + FnOnce(E) -> E) -> Element<N>
    where
        E: GenericElement<N>,
    {
        let cx = self.cx;
        self.child(view(cx, render))
    }

    pub fn child_text<A: IntoReactive<Attribute>>(self, data: A) -> Element<N> {
        let data = data.into_reactive();
        self.child_element(move |text: crate::elements::text<_>| text.data(data))
    }
}

impl<N: GenericNode> GenericComponent<N> for Element<N> {
    fn build_template(self) -> Template<N> {
        Template {
            init: self.init.into(),
            render: self.render.into(),
        }
    }
}

struct ElementInit<N>(Box<dyn FnOnce() -> N>, Box<dyn FnOnce(&N)>);

impl<N: GenericNode> ElementInit<N> {
    fn new(f: impl 'static + FnOnce() -> N) -> Self {
        Self(Box::new(f), Box::new(|_| {}))
    }

    fn child(self, child: TemplateInit<N>) -> Self {
        Self(
            self.0,
            Box::new(move |root| {
                (self.1)(root);
                child.init().append_to(root);
            }),
        )
    }
}

impl<N: GenericNode> From<ElementInit<N>> for TemplateInit<N> {
    fn from(init: ElementInit<N>) -> Self {
        TemplateInit::new(move || {
            let root = (init.0)();
            (init.1)(&root);
            View::Node(root)
        })
    }
}

type FirstChildAndOutput<N> = (Option<N>, TemplateRenderOutput<N>);

struct ElementRender<N>(
    Box<dyn FnOnce(Option<N>) -> FirstChildAndOutput<N>>,
    Box<dyn FnOnce(Option<N>) -> Option<N>>,
);

impl<N: GenericNode> ElementRender<N> {
    fn new(f: impl 'static + FnOnce(Option<N>) -> FirstChildAndOutput<N>) -> Self {
        Self(Box::new(f), Box::new(|first| first))
    }

    fn child(self, child: TemplateRender<N>) -> Self {
        Self(
            self.0,
            Box::new(move |node| {
                let node = (self.1)(node);
                child.render(node).next_sibling
            }),
        )
    }
}

impl<N: GenericNode> From<ElementRender<N>> for TemplateRender<N> {
    fn from(render: ElementRender<N>) -> Self {
        TemplateRender::new(move |root| {
            let (first_child, output) = (render.0)(root);
            let last_child = (render.1)(first_child);
            debug_assert!(last_child.is_none());
            output
        })
    }
}
