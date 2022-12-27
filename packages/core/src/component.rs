use crate::{
    node::{GenericNode, NodeType},
    template::{GlobalTemplates, TemplateContent, TemplateId},
    view::View,
};
use xframe_reactive::{untrack, Scope};

pub trait GenericComponent<N: GenericNode>: 'static + Sized {
    fn new_with_input(input: RenderInput<N>) -> Self;
    fn render_to_output(self) -> RenderOutput<N>;

    fn new(cx: Scope) -> Self {
        Self::new_with_input(RenderInput::none(cx))
    }

    fn render(self) -> View<N> {
        self.render_to_output().view
    }
}

pub struct RenderInput<N> {
    pub cx: Scope,
    mode: InputMode<N>,
}

enum InputMode<N> {
    None,
    Dehydrate,
    Hydrate { first: N, behavior: Behavior<N> },
}

#[derive(Clone)]
enum Behavior<N> {
    RemoveFrom(N),
    Nothing,
}

impl<N: GenericNode> Behavior<N> {
    pub fn apply_to(&self, node: &N) {
        match self {
            Behavior::RemoveFrom(parent) => parent.remove_child(node),
            Behavior::Nothing => {}
        }
    }
}

impl<N: GenericNode> RenderInput<N> {
    fn none(cx: Scope) -> Self {
        Self {
            cx,
            mode: InputMode::None,
        }
    }

    fn dehydrate(cx: Scope) -> Self {
        Self {
            cx,
            mode: InputMode::Dehydrate,
        }
    }

    fn hydrate(cx: Scope, first: N, behavior: Behavior<N>) -> Self {
        Self {
            cx,
            mode: InputMode::Hydrate { first, behavior },
        }
    }
}

pub struct RenderOutput<N> {
    pub view: View<N>,
    mode: OutputMode<N>,
}

enum OutputMode<N> {
    None,
    Dehydrate { dehydrated: View<N> },
    Hydrate,
}

impl<N: GenericNode> RenderOutput<N> {
    fn none(view: View<N>) -> Self {
        Self {
            view,
            mode: OutputMode::None,
        }
    }

    fn dehydrate(view: View<N>, dehydrated: View<N>) -> Self {
        Self {
            view,
            mode: OutputMode::Dehydrate { dehydrated },
        }
    }

    fn hydrate(view: View<N>) -> Self {
        Self {
            view,
            mode: OutputMode::Hydrate,
        }
    }
}

pub struct Element<N> {
    pub cx: Scope,
    mode: ElementMode<N>,
    root: N,
    dyn_view: Option<View<N>>,
}

enum ElementMode<N> {
    None,
    Dehydrate { dehydrated_root: N },
    Hydrate { last: Option<N> },
}

impl<N: GenericNode> Element<N> {
    pub fn new(cx: Scope, ty: NodeType) -> Self {
        Self::new_with_input(RenderInput::none(cx), ty)
    }

    pub fn new_with_input(input: RenderInput<N>, ty: NodeType) -> Self {
        let mode;
        let root;
        match input.mode {
            InputMode::None => {
                mode = ElementMode::None;
                root = N::create(ty);
            }
            InputMode::Dehydrate => {
                mode = ElementMode::Dehydrate {
                    dehydrated_root: N::create(ty.clone()),
                };
                root = N::create(ty);
            }
            InputMode::Hydrate { first, behavior } => {
                mode = ElementMode::Hydrate {
                    last: first.first_child(),
                };
                root = first;
                behavior.apply_to(&root);
            }
        };
        Self {
            cx: input.cx,
            mode,
            root,
            dyn_view: None,
        }
    }

    pub fn render_to_output(self) -> RenderOutput<N> {
        let view = self.dyn_view.unwrap_or_else(|| View::node(self.root));
        match self.mode {
            ElementMode::None => RenderOutput::none(view),
            ElementMode::Dehydrate { dehydrated_root } => {
                RenderOutput::dehydrate(view, View::node(dehydrated_root))
            }
            ElementMode::Hydrate { last } => {
                debug_assert!(last.is_none());
                RenderOutput::hydrate(view)
            }
        }
    }

    pub fn root(&self) -> &N {
        &self.root
    }

    pub fn is_dyn_view(&self) -> bool {
        self.dyn_view.is_some()
    }

    pub fn set_dyn_view(&mut self, mut f: impl 'static + FnMut(View<N>) -> View<N>) {
        let dyn_view = View::dyn_(self.cx, View::node(self.root.clone()));
        self.cx.create_effect({
            let dyn_view = dyn_view.clone();
            move || dyn_view.set(f(untrack(|| dyn_view.get())))
        });
        self.dyn_view = Some(dyn_view.into());
    }

    pub fn add_child<C: GenericComponent<N>>(&mut self, f: impl 'static + FnOnce(C) -> C) {
        let input = self.add_child_input();
        let output = f(C::new_with_input(input)).render_to_output();
        self.add_child_output(output);
    }

    fn add_child_input(&mut self) -> RenderInput<N> {
        match &mut self.mode {
            ElementMode::None => RenderInput::none(self.cx),
            ElementMode::Dehydrate { .. } => RenderInput::dehydrate(self.cx),
            ElementMode::Hydrate { last } => {
                let first = last.take().unwrap();
                *last = first.next_sibling();
                RenderInput::hydrate(self.cx, first, Behavior::Nothing)
            }
        }
    }

    fn add_child_output(&mut self, output: RenderOutput<N>) {
        match output.mode {
            OutputMode::None => {
                if let ElementMode::None = &self.mode {
                    output.view.append_to(&self.root);
                } else {
                    panic!("mode mismatched");
                }
            }
            OutputMode::Dehydrate { dehydrated } => {
                if let ElementMode::Dehydrate { dehydrated_root } = &self.mode {
                    output.view.append_to(&self.root);
                    dehydrated.append_to(dehydrated_root);
                } else {
                    panic!("mode mismatched");
                }
            }
            OutputMode::Hydrate => {
                if let ElementMode::Hydrate { .. } = &self.mode {
                } else {
                    panic!("mode mismatched");
                }
            }
        }
    }
}

pub struct Fragment<N> {
    pub cx: Scope,
    views: Vec<View<N>>,
    mode: FragmentMode<N>,
}

enum FragmentMode<N> {
    None,
    Dehydrate {
        dehydrated_views: Vec<View<N>>,
    },
    Hydrate {
        last: Option<N>,
        behavior: Behavior<N>,
    },
}

impl<N: GenericNode> Fragment<N> {
    pub fn new(cx: Scope) -> Self {
        Self::new_with_input(RenderInput::none(cx))
    }

    pub fn new_with_input(input: RenderInput<N>) -> Self {
        let mode = match input.mode {
            InputMode::None => FragmentMode::None,
            InputMode::Dehydrate => FragmentMode::Dehydrate {
                dehydrated_views: Vec::new(),
            },
            InputMode::Hydrate { first, behavior } => FragmentMode::Hydrate {
                last: Some(first),
                behavior,
            },
        };
        Self {
            cx: input.cx,
            views: Vec::new(),
            mode,
        }
    }

    pub fn render_to_output(self, fallback: NodeType) -> RenderOutput<N> {
        match self.mode {
            FragmentMode::None => {
                let view = if self.views.is_empty() {
                    View::node(N::create(fallback))
                } else {
                    View::fragment(self.views)
                };
                RenderOutput::none(view)
            }
            FragmentMode::Dehydrate { dehydrated_views } => {
                debug_assert_eq!(self.views.len(), dehydrated_views.len());
                let view;
                let dehydrated;
                if self.views.is_empty() {
                    view = View::node(N::create(fallback.clone()));
                    dehydrated = View::node(N::create(fallback));
                } else {
                    view = View::fragment(self.views);
                    dehydrated = View::fragment(dehydrated_views);
                }
                RenderOutput::dehydrate(view, dehydrated)
            }
            FragmentMode::Hydrate { last, .. } => {
                debug_assert!(last.is_none());
                let view = if self.views.is_empty() {
                    View::node(N::create(fallback))
                } else {
                    View::fragment(self.views)
                };
                RenderOutput::hydrate(view)
            }
        }
    }

    pub fn add_child<C: GenericComponent<N>>(&mut self, f: impl 'static + FnOnce(C) -> C) {
        let input = self.add_child_input();
        let output = f(C::new_with_input(input)).render_to_output();
        self.add_child_output(output);
    }

    fn add_child_input(&mut self) -> RenderInput<N> {
        match &mut self.mode {
            FragmentMode::None => RenderInput::none(self.cx),
            FragmentMode::Dehydrate { .. } => RenderInput::dehydrate(self.cx),
            FragmentMode::Hydrate { last, behavior } => {
                let first = last.take().unwrap();
                *last = first.next_sibling();
                RenderInput::hydrate(self.cx, first, behavior.clone())
            }
        }
    }

    fn add_child_output(&mut self, output: RenderOutput<N>) {
        match output.mode {
            OutputMode::None => {
                if let FragmentMode::None = &self.mode {
                } else {
                    panic!("mode mismatched");
                }
            }
            OutputMode::Dehydrate { dehydrated } => {
                if let FragmentMode::Dehydrate { dehydrated_views } = &mut self.mode {
                    dehydrated_views.push(dehydrated);
                } else {
                    panic!("mode mismatched");
                }
            }
            OutputMode::Hydrate => {
                if let FragmentMode::Hydrate { .. } = &self.mode {
                } else {
                    panic!("mode mismatched");
                }
            }
        }
        self.views.push(output.view);
    }
}

pub struct Root<N> {
    id: Option<fn() -> TemplateId>,
    input: RenderInput<N>,
    inner: Box<dyn FnOnce(RenderInput<N>) -> RenderOutput<N>>,
}

impl<N: GenericNode> Root<N> {
    pub fn new(cx: Scope, f: impl 'static + FnOnce(RenderInput<N>) -> RenderOutput<N>) -> Self {
        Self::new_with_input(RenderInput::none(cx), f)
    }

    pub fn new_with_input(
        input: RenderInput<N>,
        f: impl 'static + FnOnce(RenderInput<N>) -> RenderOutput<N>,
    ) -> Self {
        Self {
            id: None,
            input,
            inner: Box::new(f),
        }
    }

    pub fn render_to_output(self) -> RenderOutput<N> {
        let Self {
            id,
            input: RenderInput { cx, mode },
            inner,
        } = self;
        if let InputMode::Hydrate { .. } = &mode {
            inner(RenderInput { cx, mode })
        } else if let Some(id) = id {
            let id = id();
            let input_mode = mode;
            match GlobalTemplates::<N>::get(id) {
                Some(TemplateContent {
                    container,
                    dehydrated,
                }) => {
                    let container = container.deep_clone();
                    let RenderOutput { view, mode } = inner(RenderInput::hydrate(
                        cx,
                        container.first_child().unwrap(),
                        Behavior::RemoveFrom(container),
                    ));
                    if let OutputMode::Hydrate = mode {
                        match input_mode {
                            InputMode::None => RenderOutput::none(view),
                            InputMode::Dehydrate => {
                                RenderOutput::dehydrate(view, dehydrated.deep_clone())
                            }
                            _ => unreachable!(),
                        }
                    } else {
                        panic!("mode mismatched")
                    }
                }
                None => {
                    let RenderOutput { view, mode } = inner(RenderInput::dehydrate(cx));
                    if let OutputMode::Dehydrate { dehydrated } = mode {
                        let output = match input_mode {
                            InputMode::None => RenderOutput::none(view),
                            InputMode::Dehydrate => {
                                RenderOutput::dehydrate(view, dehydrated.deep_clone())
                            }
                            _ => unreachable!(),
                        };
                        GlobalTemplates::set(
                            id,
                            TemplateContent {
                                container: {
                                    let container = N::create(NodeType::Template(id.data().into()));
                                    dehydrated.append_to(&container);
                                    container
                                },
                                dehydrated,
                            },
                        );
                        output
                    } else {
                        panic!("mode mismatched")
                    }
                }
            }
        } else {
            inner(RenderInput { cx, mode })
        }
    }
}

impl<N: GenericNode> Root<N> {
    pub fn has_id(&self) -> bool {
        self.id.is_some()
    }

    pub fn set_id(&mut self, id: fn() -> TemplateId) {
        self.id = Some(id);
    }
}

macro_rules! impl_for_tuples {
    ($(($($Tn:ident),+),)*) => {
        #[allow(clippy::all)]
        const _: () = {
            $(impl_for_tuples!($($Tn),*);)*
        };
    };
    ($($Tn:ident),+) => {
        #[allow(non_snake_case)]
        impl<N, $($Tn,)*> GenericComponent<N> for ($($Tn,)*)
        where
            N: GenericNode,
            $($Tn: GenericComponent<N>,)*
        {
            fn new_with_input(input: RenderInput<N>) -> Self {
                enum Mode<N> {
                    None,
                    Dehydrate,
                    Hydrate { last: Option<N>, behavior: Behavior<N> },
                }

                let RenderInput { cx, mode } = input;
                let mut mode = match mode {
                    InputMode::None => Mode::None,
                    InputMode::Dehydrate => Mode::Dehydrate,
                    InputMode::Hydrate { first, behavior } => Mode::Hydrate { last: Some(first), behavior },
                };
                $(
                    let input = match &mut mode {
                        Mode::None => RenderInput::none(cx),
                        Mode::Dehydrate => RenderInput::dehydrate(cx),
                        Mode::Hydrate { last, behavior } => {
                            let first = last.take().unwrap();
                            *last = first.next_sibling();
                            RenderInput::hydrate(cx, first, behavior.clone())
                        }
                    };
                    let $Tn = $Tn::new_with_input(input);
                )*
                ($($Tn,)*)
            }

            fn render_to_output(self) -> RenderOutput<N> {
                impl_for_tuples!(@render_to_output self, $($Tn,)*)
            }
        }
    };
    (@render_to_output $val:expr, $T1:ident, $($Tn:ident,)*) => {{
        #![allow(unused_mut)]

        enum Mode<N> {
            None,
            Dehydrate { dehydrated_views: Vec<View<N>> },
            Hydrate,
        }

        let count = impl_for_tuples!(@count $($Tn,)*);
        let mut views = Vec::with_capacity(count);
        let ($T1, $($Tn,)*) = $val;
        let RenderOutput { view, mode } = $T1.render_to_output();
        let mut output_mode = match mode {
            OutputMode::None => Mode::None,
            OutputMode::Dehydrate { dehydrated } => {
                let mut dehydrated_views = Vec::with_capacity(count);
                dehydrated_views.push(dehydrated);
                Mode::Dehydrate { dehydrated_views }
            }
            OutputMode::Hydrate => Mode::Hydrate,
        };
        views.push(view);
        $(
            let RenderOutput { view, mode } = $Tn.render_to_output();
            let m = &mut output_mode;
            match mode {
                OutputMode::None => {
                    if let Mode::None = m {
                    } else {
                        panic!("mode mismatched");
                    }
                }
                OutputMode::Dehydrate { dehydrated } => {
                    if let Mode::Dehydrate { dehydrated_views } = m {
                        dehydrated_views.push(dehydrated);
                    } else {
                        panic!("mode mismatched");
                    }
                }
                OutputMode::Hydrate => {
                    if let Mode::Hydrate = m {
                    } else {
                        panic!("mode mismatched");
                    }
                }
            }
            views.push(view);
        )*
        let view = View::fragment(views);
        match output_mode {
            Mode::None => RenderOutput::none(view),
            Mode::Dehydrate { dehydrated_views } => {
                RenderOutput::dehydrate(view, View::fragment(dehydrated_views))
            }
            Mode::Hydrate => RenderOutput::hydrate(view),
        }
    }};
    (@count) => { 0 };
    (@count $T1:ident, $($Tn:ident,)*) => { 1 + impl_for_tuples!(@count $($Tn,)*) };
}

impl_for_tuples!(
    (T1),
    (T1, T2),
    (T1, T2, T3),
    (T1, T2, T3, T4),
    (T1, T2, T3, T4, T5),
    (T1, T2, T3, T4, T5, T6),
    (T1, T2, T3, T4, T5, T6, T7),
    (T1, T2, T3, T4, T5, T6, T7, T8),
    (T1, T2, T3, T4, T5, T6, T7, T8, T9),
    (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10),
    (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11),
    (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12),
    (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13),
    (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14),
    (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15),
    (T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16),
);
