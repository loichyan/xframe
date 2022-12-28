use crate::GenericNode;
use xframe_core::{Attribute, GenericComponent, IntoReactive, TemplateId};
use xframe_reactive::Scope;
use xframe_web::{elements::text, Fragment, Root};

pub fn view_root<N, C>(
    cx: Scope,
    id: fn() -> TemplateId,
    component: impl 'static + FnOnce(C) -> C,
) -> Root<N>
where
    N: GenericNode,
    C: GenericComponent<N>,
{
    Root(cx).id(id).child(component)
}

pub fn view_element<N, E>(
    _cx: Scope,
    _marker: fn(Scope) -> E,
    props: impl 'static + FnOnce(E) -> E,
    children: impl 'static + FnOnce(E) -> E,
) -> impl 'static + FnOnce(E) -> E
where
    N: GenericNode,
    E: GenericComponent<N>,
{
    move |t| children(props(t))
}

pub fn view_text<N, V: IntoReactive<Attribute>>(
    cx: Scope,
    data: V,
) -> impl 'static + FnOnce(text<N>) -> text<N>
where
    N: GenericNode,
{
    let data = data.into_reactive(cx);
    move |t| t.data(data)
}

pub fn view_component<C>(
    _cx: Scope,
    _marker: fn(Scope) -> C,
    props: impl 'static + FnOnce(C) -> C,
    children: impl 'static + FnOnce(C) -> C,
) -> impl 'static + FnOnce(C) -> C {
    move |t| children(props(t))
}

pub fn view_fragment<N>(
    _cx: Scope,
    children: impl 'static + FnOnce(Fragment<N>) -> Fragment<N>,
) -> impl 'static + FnOnce(Fragment<N>) -> Fragment<N>
where
    N: GenericNode,
{
    children
}
