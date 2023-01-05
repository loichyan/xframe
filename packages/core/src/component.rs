use crate::{
    is_debug,
    node::{GenericNode, NodeType},
    template::{GlobalState, Template},
    view::View,
    Attribute, CowStr, EventHandler, Reactive, TemplateId,
};
use xframe_reactive::Scope;

pub trait GenericComponent<N: GenericNode>: 'static + Sized {
    fn render(self) -> RenderOutput<N>;

    fn render_view(self) -> View<N> {
        self.render().view
    }
}

fn take_mode<N: GenericNode>() -> Mode<N> {
    GlobalState::mode(|global| std::mem::replace(&mut global.mode, Mode::None))
}

fn set_mode<N: GenericNode>(mode: Mode<N>) {
    GlobalState::mode(|global| global.mode = mode);
}

fn check_mode<N: GenericNode>(f: impl FnOnce(&Mode<N>)) {
    GlobalState::mode(|global| f(&global.mode));
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
        let mode;
        let root;
        match take_mode::<N>() {
            Mode::None => {
                mode = ElementMode::None;
                root = N::create(ty);
            }
            Mode::Dehydrate => {
                mode = ElementMode::Dehydrate {
                    dehydrated_root: N::create(ty.clone()),
                };
                root = N::create(ty);
            }
            Mode::Hydrate { first, behavior } => {
                mode = ElementMode::Hydrate {
                    last: first.first_child(),
                };
                root = first;
                behavior.apply_to(&root);
            }
        }
        Self {
            cx,
            mode,
            root,
            dyn_view: None,
        }
    }

    pub fn render(self) -> RenderOutput<N> {
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

    fn with_root_static(&self, f: impl Fn(&N)) {
        match &self.mode {
            ElementMode::Dehydrate { dehydrated_root } => f(dehydrated_root),
            ElementMode::Hydrate { .. } => return,
            _ => (),
        }
        f(&self.root);
    }

    pub fn set_property(&self, name: CowStr, val: Reactive<Attribute>) {
        match val {
            Reactive::Static(lit) => {
                self.with_root_static(|root| root.set_property(name.clone(), lit.clone()));
            }
            Reactive::Variable(val) => {
                self.root.set_property(name.clone(), val);
            }
            Reactive::Fn(f) => {
                let node = self.root.clone();
                self.cx
                    .create_effect(move || node.set_property(name.clone(), f()));
            }
        }
    }

    pub fn set_attribute(&self, name: CowStr, val: Reactive<Attribute>) {
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

    pub fn set_class(&self, name: CowStr, toggle: Reactive<bool>) {
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

    pub fn set_inner_text(&self, data: Reactive<Attribute>) {
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

    pub fn listen_event(&self, event: CowStr, handler: EventHandler<N::Event>) {
        self.root.listen_event(event, handler);
    }

    pub fn root(&self) -> &N {
        &self.root
    }

    pub fn is_dyn_view(&self) -> bool {
        self.dyn_view.is_some()
    }

    pub fn set_dyn_view(&mut self, mut f: impl 'static + FnMut(View<N>) -> View<N>) {
        let dyn_view = self.cx.create_signal(View::node(self.root.clone()));
        self.cx.create_effect({
            move || {
                let mut updated = false;
                dyn_view.write_slient(|current| {
                    let new = f(current.clone());
                    if !current.ref_eq(&new) {
                        *current = new;
                        updated = true;
                    }
                });
                if updated {
                    dyn_view.trigger();
                }
            }
        });
        self.dyn_view = Some(View::dyn_(dyn_view.into()));
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
            ElementMode::Hydrate { last } => {
                let first = last.take().unwrap();
                *last = first.next_sibling();
                Mode::hydrate(first, Behavior::Nothing)
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
            OutputMode::Hydrate => {
                let ElementMode::Hydrate { .. } = &self.mode else {
                    panic!("mode mismatched");
                };
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
        let mode = match take_mode::<N>() {
            Mode::None => FragmentMode::None,
            Mode::Dehydrate => FragmentMode::Dehydrate {
                dehydrated_views: Vec::new(),
            },
            Mode::Hydrate { first, behavior } => FragmentMode::Hydrate {
                last: Some(first),
                behavior,
            },
        };
        Self {
            cx,
            views: Vec::new(),
            mode,
        }
    }

    pub fn render(self, fallback: NodeType) -> RenderOutput<N> {
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
                let first = last.take().unwrap();
                *last = first.next_sibling();
                Mode::hydrate(first, behavior.clone())
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
            OutputMode::Hydrate => {
                let FragmentMode::Hydrate { .. } = &self.mode else {
                    panic!("mode mismatched");
                };
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

    pub fn render(self) -> RenderOutput<N> {
        let output = self.render_impl();
        if is_debug!() {
            check_mode::<N>(|mode| assert_eq!(mode, &Mode::none()));
        }
        output
    }

    fn render_impl(self) -> RenderOutput<N> {
        let Self { id, inner, .. } = self;
        let Some(id) = id else {
            return inner();
        };
        let mode = take_mode::<N>();
        if let Mode::Hydrate { .. } = &mode {
            set_mode(mode);
            return inner();
        }
        let id = id();
        let prev_mode = mode;
        match GlobalState::<N>::get_template(id) {
            Some(Template {
                container,
                dehydrated,
            }) => {
                let container = container.deep_clone();
                set_mode(Mode::hydrate(
                    container.first_child().unwrap(),
                    Behavior::RemoveFrom(container),
                ));
                let RenderOutput { view, mode } = inner();
                let OutputMode::Hydrate = mode else {
                    panic!("mode mismatched");
                };
                match prev_mode {
                    Mode::None => RenderOutput::none(view),
                    Mode::Dehydrate => RenderOutput::dehydrate(view, dehydrated.deep_clone()),
                    _ => unreachable!(),
                }
            }
            None => {
                set_mode(Mode::<N>::dehydrate());
                let RenderOutput { view, mode } = inner();
                let OutputMode::Dehydrate { dehydrated } = mode else {
                    panic!("mode mismatched");
                };
                let output = match prev_mode {
                    Mode::None => RenderOutput::none(view),
                    Mode::Dehydrate => RenderOutput::dehydrate(view, dehydrated.deep_clone()),
                    _ => unreachable!(),
                };
                GlobalState::set_template(
                    id,
                    Template {
                        container: {
                            let container = N::create(NodeType::Template(id.data().into()));
                            dehydrated.append_to(&container);
                            container
                        },
                        dehydrated,
                    },
                );
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
