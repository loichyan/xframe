use crate::Element;
use xframe_core::{
    is_debug, view::ViewParentExt, GenericComponent, GenericNode, IntoReactive, Reactive,
    RenderInput, RenderOutput, Value, View,
};
use xframe_reactive::{untrack, Scope};

define_placeholder!(Placeholder("PLACEHOLDER FOR `xframe::Switch` COMPONENT"));

#[allow(non_snake_case)]
pub fn Switch<N: GenericNode>(cx: Scope) -> Switch<N> {
    Switch::new(cx)
}

pub struct Switch<N> {
    input: RenderInput<N>,
    children: Vec<Branch<N>>,
}

pub struct Branch<N> {
    cond: Reactive<bool>,
    content: LazyRender<N>,
}

pub trait SwitchChild<N: GenericNode> {
    fn new(cx: Scope) -> Self;
    fn into_branch(self) -> Branch<N>;
}

impl<N: GenericNode> GenericComponent<N> for Switch<N> {
    fn new_with_input(input: RenderInput<N>) -> Self {
        Self {
            input,
            children: Vec::new(),
        }
    }

    fn render_to_output(self) -> RenderOutput<N> {
        let Self { input, children } = self;
        let cx = input.cx;
        let mut branches = children;
        let mut current_index = None;
        let mut placeholder = None;
        Element::<_, Placeholder<_>>::new_with_input(input)
            .dyn_view(move |current_view| {
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
                    return current_view;
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
                    new_view
                })
            })
            .render_to_output()
    }
}

impl<N> Switch<N>
where
    N: GenericNode,
{
    pub fn build(self) -> Self {
        self
    }

    pub fn child<C: SwitchChild<N>>(mut self, child: impl 'static + FnOnce(C) -> C) -> Switch<N> {
        self.children
            .push(child(C::new(self.input.cx)).into_branch());
        self
    }
}

#[allow(non_snake_case)]
pub fn If<N: GenericNode>(cx: Scope) -> If<N> {
    GenericComponent::new(cx)
}

pub struct If<N> {
    cx: Scope,
    input: Option<RenderInput<N>>,
    when: Option<Reactive<bool>>,
    children: Option<LazyRender<N>>,
}

impl<N: GenericNode> GenericComponent<N> for If<N> {
    fn new_with_input(input: RenderInput<N>) -> Self {
        Self {
            cx: input.cx,
            input: Some(input),
            when: None,
            children: None,
        }
    }

    fn render_to_output(mut self) -> RenderOutput<N> {
        Switch::new_with_input(self.input.take().unwrap())
            .child(|_| self)
            .render_to_output()
    }
}

impl<N: GenericNode> SwitchChild<N> for If<N> {
    fn new(cx: Scope) -> Self {
        GenericComponent::new(cx)
    }

    fn into_branch(self) -> Branch<N> {
        Branch {
            cond: self.when.expect("`If::when` was not specified"),
            content: self.children.expect("`If::child` was not specified"),
        }
    }
}

impl<N: GenericNode> If<N> {
    pub fn when<T: IntoReactive<bool>>(mut self, when: T) -> If<N> {
        if self.when.is_some() {
            panic!("`If::when` has been specified");
        }
        self.when = Some(when.into_reactive(self.cx));
        self
    }

    pub fn child<C: GenericComponent<N>>(mut self, child: impl 'static + FnOnce(C) -> C) -> If<N> {
        if self.children.is_some() {
            panic!("`If::child` has been specified");
        }
        self.children = Some(LazyRender::new(child));
        self
    }
}

#[allow(non_snake_case)]
pub fn Else<N: GenericNode>(cx: Scope) -> Else<N> {
    GenericComponent::new(cx)
}

pub struct Else<N> {
    input: Option<RenderInput<N>>,
    children: Option<LazyRender<N>>,
}

impl<N: GenericNode> GenericComponent<N> for Else<N> {
    fn new_with_input(input: RenderInput<N>) -> Self {
        Self {
            input: Some(input),
            children: None,
        }
    }

    fn render_to_output(mut self) -> RenderOutput<N> {
        if is_debug!() {
            panic!("`Else` should only be used within `Switch`");
        }
        Switch::new_with_input(self.input.take().unwrap())
            .child(|_| self)
            .render_to_output()
    }
}

impl<N: GenericNode> SwitchChild<N> for Else<N> {
    fn new(cx: Scope) -> Self {
        GenericComponent::new(cx)
    }

    fn into_branch(self) -> Branch<N> {
        Branch {
            cond: Value(true),
            content: self.children.expect("`Else::child` was not specified"),
        }
    }
}

impl<N: GenericNode> Else<N> {
    pub fn child<C: GenericComponent<N>>(
        mut self,
        child: impl 'static + FnOnce(C) -> C,
    ) -> Else<N> {
        if self.children.is_some() {
            panic!("`Else::child` has been specified");
        }
        self.children = Some(LazyRender::new(child));
        self
    }
}

struct LazyRender<N> {
    component: Option<Box<dyn FnOnce(Scope) -> View<N>>>,
    view: Option<View<N>>,
}

impl<N: GenericNode> LazyRender<N> {
    fn new<C: GenericComponent<N>>(f: impl 'static + FnOnce(C) -> C) -> Self {
        Self {
            component: Some(Box::new(move |cx| f(C::new(cx)).render())),
            view: None,
        }
    }

    fn render(&mut self, cx: Scope) -> View<N> {
        self.view
            .get_or_insert_with(|| self.component.take().unwrap()(cx))
            .clone()
    }
}
