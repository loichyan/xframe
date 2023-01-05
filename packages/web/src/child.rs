use crate::elements::text;
use xframe_core::{
    component::Fragment, GenericComponent, GenericNode, Reactive, RenderOutput, StringLike,
};
use xframe_reactive::{ReadSignal, Scope, Signal};

pub trait GenericChild<N: GenericNode>: 'static {
    fn render(self, cx: Scope) -> RenderOutput<N>;
}

impl<N, C, F> GenericChild<N> for F
where
    N: GenericNode,
    C: GenericComponent<N>,
    F: 'static + FnOnce() -> C,
{
    fn render(self, _: Scope) -> RenderOutput<N> {
        self().render()
    }
}

impl<N, T> GenericChild<N> for Signal<T>
where
    N: GenericNode,
    T: 'static + Clone + Into<StringLike>,
{
    fn render(self, cx: Scope) -> RenderOutput<N> {
        text(cx).data(self).render()
    }
}

impl<N, T> GenericChild<N> for ReadSignal<T>
where
    N: GenericNode,
    T: 'static + Clone + Into<StringLike>,
{
    fn render(self, cx: Scope) -> RenderOutput<N> {
        text(cx).data(self).render()
    }
}

impl<N, T> GenericChild<N> for Reactive<T>
where
    N: GenericNode,
    T: 'static + Into<StringLike>,
{
    fn render(self, cx: Scope) -> RenderOutput<N> {
        text(cx).data(self.cast()).render()
    }
}

macro_rules! impl_for_tuples {
    ($(($($Tn:ident),+))*) => {
        #[allow(clippy::all)]
        const _: () = {
            $(impl_for_tuples!($($Tn),*);)*
        };
    };
    ($($Tn:ident),+) => {
        impl<Node, $($Tn,)*> GenericChild<Node> for ($($Tn,)*)
        where
            Node: GenericNode,
            $($Tn: GenericChild<Node>,)*
        {
            #[allow(non_snake_case)]
            fn render(self, cx: Scope) -> RenderOutput<Node> {
                let count = impl_for_tuples!(@count $($Tn,)*);
                let ($($Tn,)*) = self;
                let mut fragment = Fragment::with_capacity(cx, count);
                $(fragment.add_child(move || $Tn.render(cx));)*
                fragment.render(|| unreachable!())
            }
        }
    };
    (@count $T1:ident, $($Tn:ident,)*) => { 1 + impl_for_tuples!(@count $($Tn,)*) };
    (@count) => { 0 };
}

impl_for_tuples! {
    (A)
    (A, B)
    (A, B, C)
    (A, B, C, D)
    (A, B, C, D, E)
    (A, B, C, D, E, F)
    (A, B, C, D, E, F, G)
    (A, B, C, D, E, F, G, H)
    (A, B, C, D, E, F, G, H, I)
    (A, B, C, D, E, F, G, H, I, J)
    (A, B, C, D, E, F, G, H, I, J, K)
    (A, B, C, D, E, F, G, H, I, J, K, L)
    (A, B, C, D, E, F, G, H, I, J, K, L, M)
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N)
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O)
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P)
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q)
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R)
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S)
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T)
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U)
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V)
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W)
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X)
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y)
    (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z)
}
