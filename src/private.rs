use crate::GenericNode;
use xframe_core::{Attribute, GenericElement, IntoReactive};
use xframe_reactive::Scope;
use xframe_web::Element;

pub fn view_element<N, E>(
    cx: Scope,
    create: fn(Scope) -> E,
    props: impl 'static + FnOnce(E) -> E,
    children: impl 'static + FnOnce(Element<N, E>) -> Element<N, E>,
) -> Element<N, E>
where
    N: GenericNode,
    E: GenericElement<N>,
{
    let _ = create;
    let element = Element(cx).with(props);
    children(element)
}

pub fn view_text<N, V: IntoReactive<Attribute>>(
    cx: Scope,
    data: V,
) -> Element<N, crate::element::text<N>>
where
    N: GenericNode,
{
    let reactive = data.into_reactive();
    Element(cx).with(|t: crate::element::text<N>| t.data(reactive))
}

pub fn view_component<Init, U1, U2, Final>(
    cx: Scope,
    create: fn(Scope) -> Init,
    props: impl 'static + FnOnce(Init) -> U1,
    children: impl 'static + FnOnce(U1) -> U2,
    build: impl 'static + FnOnce(U2) -> Final,
) -> Final {
    let component = create(cx);
    let u1 = props(component);
    let u2 = children(u1);
    build(u2)
}
