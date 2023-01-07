use crate::child::GenericChild;
use crate::components::Root;
use smallvec::SmallVec;
use xframe_core::{
    is_debug, view::ViewParentExt, GenericComponent, GenericNode, IntoReactive, Reactive,
    RenderOutput, TemplateId, View,
};
use xframe_reactive::{untrack, Scope};

define_placeholder!(struct Placeholder("PLACEHOLDER FOR `xframe::Switch` COMPONENT"));

#[allow(non_snake_case)]
pub fn Switch<N: GenericNode>(cx: Scope) -> Switch<N> {
    Switch {
        cx,
        children: SmallVec::new(),
    }
}

// TODO: conficts with svg switch element
pub struct Switch<N> {
    cx: Scope,
    children: SmallVec<[Branch<N>; 2]>,
}

pub struct Branch<N> {
    cond: Reactive<bool>,
    content: LazyRender<N>,
}

pub trait SwitchChild<N: GenericNode> {
    fn into_branch(self) -> Branch<N>;
}

impl<N: GenericNode> GenericComponent<N> for Switch<N> {
    fn render(self) -> RenderOutput<N> {
        let Self { cx, children } = self;
        let mut branches = children;
        let mut current_index = None;
        let mut placeholder = None;
        Placeholder::<N>::new(cx).render_with(move |current_view| {
            // The initial view should be the placeholder node.
            let placeholder = &*placeholder.get_or_insert_with(|| current_view.clone());
            let mut new_index = None;
            for (index, branch) in branches.iter().enumerate() {
                // Only `cond` need to be tracked.
                if branch.cond.clone().into_value() {
                    new_index = Some(index);
                    break;
                }
            }
            if new_index == current_index {
                return None;
            }
            current_index = new_index;
            untrack(|| {
                let parent = current_view.parent();
                let new_view = new_index
                    .map(|i| branches[i].content.render(cx))
                    // Fallback to a placeholder.
                    .unwrap_or_else(|| placeholder.clone());
                parent.replace_child(&new_view, &current_view);
                debug_assert!(new_view.check_mount_order());
                Some(new_view)
            })
        })
    }
}

impl<N> Switch<N>
where
    N: GenericNode,
{
    pub fn build(self) -> Self {
        self
    }

    pub fn child<C: SwitchChild<N>>(mut self, child: impl 'static + FnOnce() -> C) -> Switch<N> {
        self.children.push(child().into_branch());
        self
    }
}

#[allow(non_snake_case)]
pub fn If<N: GenericNode>(cx: Scope) -> If<N> {
    If {
        cx,
        id: None,
        when: None,
        children: None,
    }
}

pub struct If<N> {
    cx: Scope,
    id: Option<fn() -> TemplateId>,
    when: Option<Reactive<bool>>,
    children: Option<Box<dyn FnOnce() -> RenderOutput<N>>>,
}

impl<N: GenericNode> GenericComponent<N> for If<N> {
    fn render(self) -> RenderOutput<N> {
        Switch(self.cx).child(|| self).render()
    }
}

impl<N: GenericNode> SwitchChild<N> for If<N> {
    fn into_branch(self) -> Branch<N> {
        Branch {
            cond: self.when.expect("`If::when` was not specified"),
            content: LazyRender::new(
                self.id,
                self.children.expect("`If::child` was not specified"),
            ),
        }
    }
}

impl<N: GenericNode> If<N> {
    pub fn id(mut self, id: fn() -> TemplateId) -> Self {
        if self.id.is_some() {
            panic!("`If::id` has been specified");
        }
        self.id = Some(id);
        self
    }

    pub fn when<T: IntoReactive<bool>>(mut self, when: T) -> If<N> {
        if self.when.is_some() {
            panic!("`If::when` has been specified");
        }
        self.when = Some(when.into_reactive());
        self
    }

    pub fn child(mut self, child: impl GenericChild<N>) -> If<N> {
        if self.children.is_some() {
            panic!("`If::child` has been specified");
        }
        let cx = self.cx;
        self.children = Some(Box::new(move || child.render(cx)));
        self
    }
}

#[allow(non_snake_case)]
pub fn Else<N: GenericNode>(cx: Scope) -> Else<N> {
    Else {
        cx,
        id: None,
        children: None,
    }
}

pub struct Else<N> {
    cx: Scope,
    id: Option<fn() -> TemplateId>,
    children: Option<Box<dyn FnOnce() -> RenderOutput<N>>>,
}

impl<N: GenericNode> GenericComponent<N> for Else<N> {
    fn render(self) -> RenderOutput<N> {
        if is_debug!() {
            panic!("`Else` should only be used within `Switch`");
        }
        Switch(self.cx).child(|| self).render()
    }
}

impl<N: GenericNode> SwitchChild<N> for Else<N> {
    fn into_branch(self) -> Branch<N> {
        Branch {
            cond: Reactive::Static(true),
            content: LazyRender::new(
                self.id,
                self.children.expect("`Else::child` was not specified"),
            ),
        }
    }
}

impl<N: GenericNode> Else<N> {
    pub fn id(mut self, id: fn() -> TemplateId) -> Self {
        if self.id.is_some() {
            panic!("`Else::id` has been specified");
        }
        self.id = Some(id);
        self
    }

    pub fn child(mut self, child: impl GenericChild<N>) -> Else<N> {
        if self.children.is_some() {
            panic!("`Else::child` has been specified");
        }
        let cx = self.cx;
        self.children = Some(Box::new(move || child.render(cx)));
        self
    }
}

struct LazyRender<N> {
    component: Option<Box<dyn FnOnce(Scope) -> View<N>>>,
    view: Option<View<N>>,
}

impl<N: GenericNode> LazyRender<N> {
    fn new(id: Option<fn() -> TemplateId>, f: impl 'static + FnOnce() -> RenderOutput<N>) -> Self {
        Self {
            component: Some(Box::new(move |cx| {
                if let Some(id) = id {
                    Root(cx).id(id).child(f).render_view()
                } else {
                    f().render_view()
                }
            })),
            view: None,
        }
    }

    fn render(&mut self, cx: Scope) -> View<N> {
        self.view
            .get_or_insert_with(|| self.component.take().unwrap()(cx))
            .clone()
    }
}
