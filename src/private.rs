use crate::GenericNode;
use xframe_core::{GenericComponent, IntoReactive, Reactive, StringLike, TemplateId};
use xframe_reactive::Scope;
use xframe_web::{elements::text, Fragment, GenericElement, Root};

pub fn view_root<N, C>(
    cx: Scope,
    id: fn() -> TemplateId,
    component: impl 'static + FnOnce() -> C,
) -> Root<N>
where
    N: GenericNode,
    C: GenericComponent<N>,
{
    Root(cx).id(id).child(component)
}

pub fn view_element<N, E>(
    cx: Scope,
    new: fn(Scope) -> E,
    props: impl 'static + FnOnce(E) -> E,
    children: impl 'static + FnOnce(E) -> E,
) -> impl 'static + FnOnce() -> E
where
    N: GenericNode,
    E: GenericElement<N>,
{
    move || children(props(new(cx)))
}

pub fn view_text<N, V: IntoReactive<StringLike>>(
    cx: Scope,
    data: V,
) -> impl 'static + FnOnce() -> text<N>
where
    N: GenericNode,
{
    let data = data.into_reactive();
    move || text(cx).data(data)
}

pub fn view_text_literal<N>(cx: Scope, data: &'static str) -> impl 'static + FnOnce() -> text<N>
where
    N: GenericNode,
{
    move || text(cx).data(Reactive::Static(StringLike::Literal(data)))
}

pub fn view_component<C: 'static>(
    cx: Scope,
    new: fn(Scope) -> C,
    props: impl 'static + FnOnce(C) -> C,
    children: impl 'static + FnOnce(C) -> C,
) -> impl 'static + FnOnce() -> C {
    move || children(props(new(cx)))
}

pub fn view_fragment<N>(
    cx: Scope,
    count: usize,
    children: impl 'static + FnOnce(Fragment<N>) -> Fragment<N>,
) -> impl 'static + FnOnce() -> Fragment<N>
where
    N: GenericNode,
{
    move || children(Fragment::with_capacity(cx, count))
}
