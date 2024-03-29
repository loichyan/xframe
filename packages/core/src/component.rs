use crate::{
    event::EventHandler,
    node::{GenericNode, NodeType},
    reactive::Reactive,
    str::{RcStr, StringLike},
    template::{GlobalStateInner, Template, TemplateId},
    view::View,
};
use xframe_reactive::Scope;

pub trait GenericComponent<N: GenericNode>: 'static + Sized {
    fn render(self) -> RenderOutput<N>;

    fn render_view(self) -> View<N> {
        self.render().view
    }
}

impl<N: GenericNode> GenericComponent<N> for RenderOutput<N> {
    fn render(self) -> RenderOutput<N> {
        self
    }
}

fn take_mode<N: GenericNode>() -> Mode<N> {
    GlobalStateInner::mode(|global| std::mem::replace(&mut global.mode, Mode::None))
}

fn set_mode<N: GenericNode>(mode: Mode<N>) {
    GlobalStateInner::mode(|global| global.mode = mode);
}

fn check_mode<N: GenericNode>(f: impl FnOnce(&Mode<N>)) {
    GlobalStateInner::mode(|global| f(&global.mode));
}

pub(crate) struct GlobalMode<N> {
    mode: Mode<N>,
}

impl<N: GenericNode> Default for GlobalMode<N> {
    fn default() -> Self {
        Self { mode: Mode::None }
    }
}

#[derive(Debug, Eq, PartialEq)]
enum Mode<N> {
    None,
    Dehydrate,
    Hydrate { first: N, behavior: Behavior<N> },
}

impl<N: GenericNode> Mode<N> {
    fn none() -> Self {
        Mode::None
    }

    fn dehydrate() -> Self {
        Mode::Dehydrate
    }

    fn hydrate(first: N, behavior: Behavior<N>) -> Self {
        Mode::Hydrate { first, behavior }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
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

pub struct RenderOutput<N> {
    view: View<N>,
    mode: OutputMode<N>,
}

enum OutputMode<N> {
    None,
    Dehydrate { dehydrated: View<N> },
    Hydrate { next: Option<N> },
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

    fn hydrate(view: View<N>, next: Option<N>) -> Self {
        Self {
            view,
            mode: OutputMode::Hydrate { next },
        }
    }
}

pub struct Element<N> {
    pub cx: Scope,
    mode: ElementMode<N>,
    root: N,
}

enum ElementMode<N> {
    None,
    Dehydrate { dehydrated_root: N },
    Hydrate { next: Option<N>, last: Option<N> },
}

impl<N: GenericNode> Element<N> {
    pub fn new(cx: Scope, init: fn() -> N) -> Self {
        let mode;
        let root;
        match take_mode::<N>() {
            Mode::None => {
                mode = ElementMode::None;
                root = init();
            }
            Mode::Dehydrate => {
                mode = ElementMode::Dehydrate {
                    dehydrated_root: init(),
                };
                root = init();
            }
            Mode::Hydrate { first, behavior } => {
                mode = ElementMode::Hydrate {
                    next: first.next_sibling(),
                    last: first.first_child(),
                };
                root = first;
                behavior.apply_to(&root);
            }
        }
        Self { cx, mode, root }
    }

    pub fn render(self) -> RenderOutput<N> {
        self.render_impl(None)
    }

    pub fn render_with(
        self,
        mut f: impl 'static + FnMut(View<N>) -> Option<View<N>>,
    ) -> RenderOutput<N> {
        let dyn_view = self.cx.create_signal(View::node(self.root.clone()));
        self.cx.create_effect({
            move || {
                let mut updated = false;
                dyn_view.write_slient(|current| {
                    if let Some(new) = f(current.clone()) {
                        *current = new;
                        updated = true;
                    }
                });
                if updated {
                    dyn_view.trigger();
                }
            }
        });
        self.render_impl(Some(View::dyn_(dyn_view.into())))
    }

    fn render_impl(self, dyn_view: Option<View<N>>) -> RenderOutput<N> {
        let view = dyn_view.unwrap_or_else(|| View::node(self.root));
        match self.mode {
            ElementMode::None => RenderOutput::none(view),
            ElementMode::Dehydrate { dehydrated_root } => {
                RenderOutput::dehydrate(view, View::node(dehydrated_root))
            }
            ElementMode::Hydrate { next, last } => {
                debug_assert_eq!(last, None);
                RenderOutput::hydrate(view, next)
            }
        }
    }

    pub fn add_child<C: GenericComponent<N>>(&mut self, f: impl 'static + FnOnce() -> C) {
        self.before_child();
        let output = f().render();
        self.after_child(output);
    }

    fn before_child(&mut self) {
        let mode = match &mut self.mode {
            ElementMode::None => Mode::none(),
            ElementMode::Dehydrate { .. } => Mode::dehydrate(),
            ElementMode::Hydrate { last, .. } => {
                Mode::hydrate(last.take().unwrap(), Behavior::Nothing)
            }
        };
        set_mode(mode);
    }

    fn after_child(&mut self, output: RenderOutput<N>) {
        if is_debug!() {
            check_mode::<N>(|mode| assert_eq!(mode, &Mode::none()));
        }
        match output.mode {
            OutputMode::None => {
                let ElementMode::None = &self.mode  else {
                    panic!("mode mismatched");
                };
                output.view.append_to(&self.root);
            }
            OutputMode::Dehydrate { dehydrated } => {
                let ElementMode::Dehydrate { dehydrated_root } = &self.mode else {
                    panic!("mode mismatched");
                };
                output.view.append_to(&self.root);
                dehydrated.append_to(dehydrated_root);
            }
            OutputMode::Hydrate { next } => {
                let ElementMode::Hydrate { last, .. } = &mut self.mode else {
                    panic!("mode mismatched");
                };
                *last = next;
            }
        }
    }

    fn with_root_static(&self, f: impl Fn(&N)) {
        match &self.mode {
            ElementMode::Dehydrate { dehydrated_root } => f(dehydrated_root),
            ElementMode::Hydrate { .. } => return,
            _ => (),
        }
        f(&self.root);
    }

    pub fn root(&self) -> &N {
        &self.root
    }

    pub fn set_property(&self, name: RcStr, val: Reactive<StringLike>) {
        match val {
            Reactive::Static(lit) => {
                self.with_root_static(|root| root.set_property(name.clone(), lit.clone()));
            }
            Reactive::Variable(val) => {
                self.root.set_property(name, val);
            }
            Reactive::Fn(f) => {
                let node = self.root.clone();
                self.cx
                    .create_effect(move || node.set_property(name.clone(), f()));
            }
        }
    }

    pub fn set_attribute(&self, name: RcStr, val: Reactive<StringLike>) {
        match val {
            Reactive::Static(val) => {
                self.with_root_static(|root| root.set_attribute(name.clone(), val.clone()));
            }
            Reactive::Variable(val) => {
                self.root.set_attribute(name, val);
            }
            Reactive::Fn(f) => {
                let node = self.root.clone();
                self.cx
                    .create_effect(move || node.set_attribute(name.clone(), f()));
            }
        }
    }

    pub fn set_class(&self, name: RcStr, toggle: Reactive<bool>) {
        match toggle {
            Reactive::Static(toggle) => {
                if toggle {
                    self.with_root_static(|root| root.add_class(name.clone()));
                }
            }
            Reactive::Variable(toggle) => {
                if toggle {
                    self.root.add_class(name);
                }
            }
            Reactive::Fn(f) => {
                let node = self.root.clone();
                self.cx.create_effect(move || {
                    if f() {
                        node.add_class(name.clone());
                    } else {
                        node.remove_class(name.clone());
                    }
                });
            }
        }
    }

    pub fn set_classes(&self, names: &[&'static str]) {
        self.with_root_static(|root| {
            for &name in names {
                root.add_class(name.into())
            }
        });
    }

    pub fn set_inner_text(&self, data: Reactive<StringLike>) {
        match data {
            Reactive::Static(data) => {
                self.with_root_static(|root| root.set_inner_text(data.clone().into_string()));
            }
            Reactive::Variable(data) => {
                self.root.set_inner_text(data.into_string());
            }
            Reactive::Fn(f) => {
                let node = self.root.clone();
                self.cx
                    .create_effect(move || node.set_inner_text(f().into_string()));
            }
        }
    }

    pub fn listen_event(&self, event: RcStr, handler: EventHandler<N::Event>) {
        self.root.listen_event(event, handler);
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
        Self::with_capacity(cx, 0)
    }

    pub fn with_capacity(cx: Scope, capacity: usize) -> Self {
        let mode = match take_mode::<N>() {
            Mode::None => FragmentMode::None,
            Mode::Dehydrate => FragmentMode::Dehydrate {
                dehydrated_views: Vec::with_capacity(capacity),
            },
            Mode::Hydrate { first, behavior } => FragmentMode::Hydrate {
                last: Some(first),
                behavior,
            },
        };
        Self {
            cx,
            views: Vec::with_capacity(capacity),
            mode,
        }
    }

    pub fn render(self, fallback: fn() -> N) -> RenderOutput<N> {
        match self.mode {
            FragmentMode::None => {
                let view = if self.views.is_empty() {
                    View::node(fallback())
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
                    view = View::node(fallback());
                    dehydrated = View::node(fallback());
                } else {
                    view = View::fragment(self.views);
                    dehydrated = View::fragment(dehydrated_views);
                }
                RenderOutput::dehydrate(view, dehydrated)
            }
            FragmentMode::Hydrate { last, .. } => {
                let view = if self.views.is_empty() {
                    View::node(fallback())
                } else {
                    View::fragment(self.views)
                };
                RenderOutput::hydrate(view, last)
            }
        }
    }

    pub fn add_child<C: GenericComponent<N>>(&mut self, f: impl 'static + FnOnce() -> C) {
        self.before_child();
        let output = f().render();
        self.after_child(output);
    }

    fn before_child(&mut self) {
        let mode = match &mut self.mode {
            FragmentMode::None => Mode::none(),
            FragmentMode::Dehydrate { .. } => Mode::dehydrate(),
            FragmentMode::Hydrate { last, behavior } => {
                Mode::hydrate(last.take().unwrap(), behavior.clone())
            }
        };
        set_mode(mode);
    }

    fn after_child(&mut self, output: RenderOutput<N>) {
        if is_debug!() {
            check_mode::<N>(|mode| assert_eq!(mode, &Mode::none()));
        }
        match output.mode {
            OutputMode::None => {
                let FragmentMode::None = &self.mode else {
                    panic!("mode mismatched");
                };
            }
            OutputMode::Dehydrate { dehydrated } => {
                let FragmentMode::Dehydrate { dehydrated_views } = &mut self.mode else {
                    panic!("mode mismatched");
                };
                dehydrated_views.push(dehydrated);
            }
            OutputMode::Hydrate { next } => {
                let FragmentMode::Hydrate { last, .. } = &mut self.mode else {
                    panic!("mode mismatched");
                };
                *last = next;
            }
        }
        self.views.push(output.view);
    }
}

pub struct Root<N> {
    pub cx: Scope,
    id: Option<fn() -> TemplateId>,
    inner: Box<dyn FnOnce() -> RenderOutput<N>>,
}

impl<N: GenericNode> Root<N> {
    pub fn new(cx: Scope, f: impl 'static + FnOnce() -> RenderOutput<N>) -> Self {
        Self {
            cx,
            id: None,
            inner: Box::new(f),
        }
    }

    pub fn render(self, placeholder: fn(&'static str) -> N) -> RenderOutput<N> {
        let output = self.render_impl(placeholder);
        if is_debug!() {
            check_mode::<N>(|mode| assert_eq!(mode, &Mode::none()));
        }
        output
    }

    fn render_impl(self, placeholder: fn(&'static str) -> N) -> RenderOutput<N> {
        let Self { id, inner, .. } = self;
        let Some(id) = id else {
            return inner();
        };
        let mode = take_mode::<N>();
        let id = id();
        let prev_mode = mode;
        match GlobalStateInner::<N>::get_template(id) {
            Some(Template { container, .. }) => {
                let container = container.deep_clone();
                set_mode(Mode::hydrate(
                    container.first_child().unwrap(),
                    Behavior::RemoveFrom(container),
                ));
                let RenderOutput { view, mode } = inner();
                let OutputMode::Hydrate { next, .. } = mode else {
                    panic!("mode mismatched");
                };
                debug_assert_eq!(next, None);
                match prev_mode {
                    Mode::None => RenderOutput::none(view),
                    Mode::Dehydrate => {
                        RenderOutput::dehydrate(view, View::node(placeholder(id.data())))
                    }
                    Mode::Hydrate { first, .. } => {
                        let parent = first.parent().unwrap();
                        let next = first.next_sibling();
                        view.replace_with(&parent, &view);
                        RenderOutput::hydrate(view, next)
                    }
                }
            }
            None => {
                if let Mode::Hydrate { .. } = &prev_mode {
                    set_mode(prev_mode);
                    return inner();
                }
                set_mode(Mode::<N>::dehydrate());
                let RenderOutput { view, mode } = inner();
                let OutputMode::Dehydrate { dehydrated } = mode else {
                    panic!("mode mismatched");
                };
                let output = match prev_mode {
                    Mode::None => RenderOutput::none(view),
                    Mode::Dehydrate => {
                        RenderOutput::dehydrate(view, View::node(placeholder(id.data())))
                    }
                    _ => unreachable!(),
                };
                let container = N::create(NodeType::Template(id.data().into()));
                dehydrated.append_to(&container);
                GlobalStateInner::set_template(id, Template { container });
                output
            }
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
