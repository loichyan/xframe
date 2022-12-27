use std::marker::PhantomData;
use xframe_core::{
    component::Element as ElementBase, Attribute, GenericComponent, GenericElement, GenericNode,
    IntoReactive, RenderInput, RenderOutput, View,
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
    GenericComponent::new(cx)
}

pub struct Element<N, E> {
    inner: ElementBase<N>,
    marker: PhantomData<E>,
}

impl<N, E> GenericComponent<N> for Element<N, E>
where
    N: GenericNode,
    E: GenericElement<N>,
{
    fn new_with_input(input: RenderInput<N>) -> Self {
        Self {
            inner: ElementBase::new_with_input(input, E::TYPE),
            marker: PhantomData,
        }
    }

    fn render_to_output(self) -> RenderOutput<N> {
        self.inner.render_to_output()
    }
}

impl<N, E> Element<N, E>
where
    N: GenericNode,
    E: GenericElement<N>,
{
    pub fn then(self, then: impl FnOnce(E) -> E) -> Self {
        let node = self.inner.root().clone();
        then(E::create_with_node(self.inner.cx, node));
        self
    }

    pub fn dyn_view(mut self, dyn_view: impl 'static + FnMut(View<N>) -> View<N>) -> Self {
        if self.inner.is_dyn_view() {
            panic!("`Element::dyn_view` has been specified")
        }
        self.inner.set_dyn_view(dyn_view);
        self
    }

    pub fn child<C: GenericComponent<N>>(mut self, child: impl 'static + FnOnce(C) -> C) -> Self {
        self.inner.add_child(child);
        self
    }

    pub fn child_element<Child>(self, then: impl 'static + FnOnce(Child) -> Child) -> Self
    where
        Child: GenericElement<N>,
    {
        self.child(move |t: Element<N, Child>| t.then(then))
    }

    pub fn child_text<A: IntoReactive<Attribute>>(self, data: A) -> Self {
        let data = data.into_reactive(self.inner.cx);
        self.child_element(move |text: crate::elements::text<_>| text.data(data))
    }
}
