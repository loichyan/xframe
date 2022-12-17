use crate::view;
use xframe_core::{
    template::{Template, TemplateInit, TemplateRender, TemplateRenderOutput},
    Attribute, GenericComponent, GenericElement, GenericNode, IntoReactive, View,
};
use xframe_reactive::Scope;

define_placeholder!(Placeholder("Placeholder for `xframe::Fragment` Component"));

#[allow(non_snake_case)]
pub fn Fragment<N: GenericNode>(cx: Scope) -> Fragment<N> {
    Fragment {
        cx,
        init: FragmentInit::new(),
        render: FragmentRender::new(),
    }
}

pub struct Fragment<N> {
    cx: Scope,
    init: FragmentInit<N>,
    render: FragmentRender<N>,
}

impl<N: GenericNode> Fragment<N> {
    pub fn build(self) -> impl GenericComponent<N> {
        self
    }

    pub fn child<C>(self, child: C) -> Fragment<N>
    where
        C: GenericComponent<N>,
    {
        let child = child.build_template();
        Fragment {
            cx: self.cx,
            init: self.init.child(child.init),
            render: self.render.child(child.render),
        }
    }

    pub fn child_element<E>(self, render: impl 'static + FnOnce(E) -> E) -> Fragment<N>
    where
        E: GenericElement<N>,
    {
        let cx = self.cx;
        self.child(view(cx, render))
    }

    pub fn child_text<A: IntoReactive<Attribute>>(self, data: A) -> Fragment<N> {
        let data = data.into_reactive();
        self.child_element(move |text: crate::elements::text<_>| text.data(data))
    }
}

impl<N: GenericNode> GenericComponent<N> for Fragment<N> {
    fn build_template(self) -> Template<N> {
        Template {
            init: self.init.into(),
            render: self.render.into(),
        }
    }
}

#[allow(clippy::type_complexity)]
struct FragmentInit<N>(Box<dyn FnOnce(&mut Vec<View<N>>)>);

impl<N: GenericNode> FragmentInit<N> {
    fn new() -> Self {
        Self(Box::new(|_| {}))
    }

    fn child(self, child: TemplateInit<N>) -> Self {
        Self(Box::new(|fragment| {
            (self.0)(fragment);
            fragment.push(child.init());
        }))
    }
}

impl<N: GenericNode> From<FragmentInit<N>> for TemplateInit<N> {
    fn from(init: FragmentInit<N>) -> Self {
        TemplateInit::new(|| {
            let mut fragment = Default::default();
            (init.0)(&mut fragment);
            if fragment.is_empty() {
                // Return a placeholder.
                View::Node(Placeholder::new().into_node())
            } else {
                View::from(fragment)
            }
        })
    }
}

#[allow(clippy::type_complexity)]
struct FragmentRender<N>(Box<dyn FnOnce(Option<N>, &mut Vec<View<N>>) -> Option<N>>);

impl<N: GenericNode> FragmentRender<N> {
    fn new() -> Self {
        Self(Box::new(|first, _| first))
    }

    fn child(self, child: TemplateRender<N>) -> Self {
        Self(Box::new(|first, fragments| {
            let output = child.render((self.0)(first, fragments));
            fragments.push(output.view);
            output.next_sibling
        }))
    }
}

impl<N: GenericNode> From<FragmentRender<N>> for TemplateRender<N> {
    fn from(render: FragmentRender<N>) -> Self {
        TemplateRender::new(|node| {
            let mut fragments = Default::default();
            let next_sibling = (render.0)(node, &mut fragments);
            if fragments.is_empty() {
                // Ignore the placeholder.
                let placeholder = next_sibling.unwrap_or_else(|| unreachable!());
                TemplateRenderOutput {
                    next_sibling: placeholder.next_sibling(),
                    view: View::Node(placeholder),
                }
            } else {
                TemplateRenderOutput {
                    next_sibling,
                    view: View::from(fragments),
                }
            }
        })
    }
}
