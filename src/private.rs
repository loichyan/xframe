use crate::GenericNode;
use xframe_core::{Attribute, GenericComponent, IntoReactive, TemplateId};
use xframe_reactive::Scope;
use xframe_web::{elements::text, Fragment, GenericElement, Root};

pub fn view_root<N, C>(
    cx: Scope,
    id: fn() -> TemplateId,
    component: impl 'static + FnOnce(Scope) -> C,
) -> Root<N>
where
    N: GenericNode,
    C: GenericComponent<N>,
{
    Root(cx).id(id).child(component)
}

pub fn view_element<N, E>(
    _cx: Scope,
    new: fn(Scope) -> E,
    props: impl 'static + FnOnce(E) -> E,
    children: impl 'static + FnOnce(E) -> E,
) -> impl 'static + FnOnce(Scope) -> E
where
    N: GenericNode,
    E: GenericElement<N>,
{
    move |cx| children(props(new(cx)))
}

pub fn view_text<N, V: IntoReactive<Attribute>>(
    cx: Scope,
    data: V,
) -> impl 'static + FnOnce(Scope) -> text<N>
where
    N: GenericNode,
{
    let data = data.into_reactive(cx);
    move |cx| text(cx).data(data)
}

pub fn view_component<C: 'static>(
    _cx: Scope,
    new: fn(Scope) -> C,
    props: impl 'static + FnOnce(C) -> C,
    children: impl 'static + FnOnce(C) -> C,
) -> impl 'static + FnOnce(Scope) -> C {
    move |cx| children(props(new(cx)))
}

pub fn view_fragment<N>(
    _cx: Scope,
    children: impl 'static + FnOnce(Fragment<N>) -> Fragment<N>,
) -> impl 'static + FnOnce(Scope) -> Fragment<N>
where
    N: GenericNode,
{
    move |cx| children(Fragment(cx))
}
