use crate::view;
use std::rc::Rc;
use xframe_core::{
    component::{Template, TemplateInit, TemplateRender},
    Attribute, View, GenericComponent, GenericElement, GenericNode, IntoReactive,
};
use xframe_reactive::Scope;

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

    pub fn child_element<E, F>(self, render: F) -> Fragment<N>
    where
        E: GenericElement<N>,
        F: 'static + FnOnce(E),
    {
        let cx = self.cx;
        self.child(view(cx, render))
    }

    pub fn child_text<A: IntoReactive<Attribute>>(self, data: A) -> Fragment<N> {
        let data = data.into_reactive();
        self.child_element(move |text: crate::elements::text<_>| {
            text.data(data);
        })
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
struct FragmentInit<N>(Box<dyn FnOnce(&mut Vec<N>)>);

impl<N: GenericNode> FragmentInit<N> {
    fn new() -> Self {
        Self(Box::new(|_| {}))
    }

    fn child(self, child: TemplateInit<N>) -> Self {
        Self(Box::new(|fragments| {
            (self.0)(fragments);
            fragments.extend(child.init().iter().cloned());
        }))
    }
}

impl<N: GenericNode> From<FragmentInit<N>> for TemplateInit<N> {
    fn from(init: FragmentInit<N>) -> Self {
        TemplateInit::new(|| {
            let mut fragments = Default::default();
            (init.0)(&mut fragments);
            View::Fragment(Rc::from(fragments.into_boxed_slice()))
        })
    }
}

struct FragmentRender<N>(Box<dyn FnOnce(Option<N>) -> Option<N>>);

impl<N: GenericNode> FragmentRender<N> {
    fn new() -> Self {
        Self(Box::new(|first| first))
    }

    fn child(self, child: TemplateRender<N>) -> Self {
        Self(Box::new(|first| child.render((self.0)(first))))
    }
}

impl<N: GenericNode> From<FragmentRender<N>> for TemplateRender<N> {
    fn from(render: FragmentRender<N>) -> Self {
        TemplateRender::new(|node| (render.0)(node))
    }
}
