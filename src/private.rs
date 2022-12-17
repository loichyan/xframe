use crate::{view, GenericNode};
use xframe_core::GenericElement;
use xframe_reactive::Scope;
use xframe_web::Element;

pub fn view_builtin<N, E>(
    cx: Scope,
    _: fn(Scope) -> E,
    render: impl 'static + FnOnce(E) -> E,
) -> Element<N>
where
    N: GenericNode,
    E: GenericElement<N>,
{
    view(cx, render)
}
