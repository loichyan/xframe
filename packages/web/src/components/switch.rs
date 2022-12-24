use crate::Element;
use xframe_core::{
    component::DynComponent, is_debug, view::ViewParentExt, GenericComponent, GenericElement,
    GenericNode, IntoReactive, Reactive, Template, Value, View,
};
use xframe_reactive::{untrack, Scope};

define_placeholder!(Placeholder("PLACEHOLDER FOR `xframe::Switch` COMPONENT"));

#[allow(non_snake_case)]
pub fn Switch<N: GenericNode>(cx: Scope) -> Switch<N> {
    Switch {
        cx,
        children: Default::default(),
    }
}

pub struct Switch<N> {
    cx: Scope,
    children: Vec<SwitchChild<N>>,
}

pub struct SwitchChild<N> {
    cond: Reactive<bool>,
    content: LazyRender<N>,
}

impl<N: GenericNode> GenericComponent<N> for Switch<N> {
    fn build_template(self) -> Template<N> {
        let Self { cx, children } = self;
        Element(cx)
            .with_view(move |placeholder: Placeholder<N>| {
                let placeholder = View::node(placeholder.into_node());
                let dyn_view = View::dyn_(cx, placeholder.clone());
                cx.create_effect({
                    let dyn_view = dyn_view.clone();
                    let mut branches = children;
                    let mut current_index = None;
                    move || {
                        let mut new_index = None;
                        for (index, branch) in branches.iter().enumerate() {
                            // Only `cond` need to be tracked.
                            if branch.cond.clone().into_value() {
                                new_index = Some(index);
                                break;
                            }
                        }
                        if new_index == current_index {
                            return;
                        }
                        current_index = new_index;
                        untrack(|| {
                            let current_view = dyn_view.get();
                            let parent = current_view.parent();
                            let new_view = new_index
                                .map(|i| branches[i].content.render())
                                // Fallback to a placeholder.
                                .unwrap_or_else(|| placeholder.clone());
                            parent.replace_child(&new_view, &current_view);
                            debug_assert!(new_view.check_mount_order());
                            dyn_view.set(new_view);
                        });
                    }
                });
                View::from(dyn_view)
            })
            .build_template()
    }
}

impl<N> Switch<N>
where
    N: GenericNode,
{
    pub fn build(self) -> Self {
        self
    }

    pub fn child<C: Into<SwitchChild<N>>>(mut self, child: C) -> Switch<N> {
        self.children.push(child.into());
        self
    }
}

#[allow(non_snake_case)]
pub fn If<N: GenericNode>(cx: Scope) -> If<N> {
    If {
        cx,
        when: None,
        children: None,
    }
}

pub struct If<N> {
    cx: Scope,
    when: Option<Reactive<bool>>,
    children: Option<DynComponent<N>>,
}

impl<N: GenericNode> From<If<N>> for SwitchChild<N> {
    fn from(value: If<N>) -> Self {
        SwitchChild {
            cond: value.when.expect("`If::when` was not specified"),
            content: value
                .children
                .expect("`If::child` was not specified")
                .into(),
        }
    }
}

impl<N: GenericNode> GenericComponent<N> for If<N> {
    fn build_template(self) -> Template<N> {
        let cx = self.cx;
        Switch(cx).child(self).build_template()
    }
}

impl<N: GenericNode> If<N> {
    pub fn build(self) -> Self {
        self
    }

    pub fn when<T: IntoReactive<bool>>(mut self, when: T) -> If<N> {
        if self.when.is_some() {
            panic!("`If::when` has already been specified");
        }
        self.when = Some(when.into_reactive(self.cx));
        self
    }

    pub fn child<C: GenericComponent<N>>(mut self, child: C) -> If<N> {
        if self.children.is_some() {
            panic!("`If::child` has already been specified");
        }
        self.children = Some(child.into_dyn_component());
        self
    }
}

#[allow(non_snake_case)]
pub fn Else<N: GenericNode>(cx: Scope) -> Else<N> {
    Else { cx, children: None }
}

pub struct Else<N> {
    cx: Scope,
    children: Option<DynComponent<N>>,
}

impl<N: GenericNode> From<Else<N>> for SwitchChild<N> {
    fn from(value: Else<N>) -> Self {
        SwitchChild {
            cond: Value(true),
            content: value
                .children
                .expect("`Else::child` was not specified")
                .into(),
        }
    }
}

impl<N: GenericNode> GenericComponent<N> for Else<N> {
    fn build_template(self) -> Template<N> {
        let cx = self.cx;
        if is_debug!() {
            panic!("`Else` should only be used within `Switch`");
        }
        Switch(cx).child(self).build_template()
    }
}

impl<N: GenericNode> Else<N> {
    pub fn build(self) -> Self {
        self
    }

    pub fn child<C: GenericComponent<N>>(mut self, child: C) -> Else<N> {
        if self.children.is_some() {
            panic!("`Else::child` has already been specified");
        }
        self.children = Some(child.into_dyn_component());
        self
    }
}

struct LazyRender<N> {
    component: Option<DynComponent<N>>,
    view: Option<View<N>>,
}

impl<N: GenericNode> From<DynComponent<N>> for LazyRender<N> {
    fn from(value: DynComponent<N>) -> Self {
        Self::new(value)
    }
}

impl<N: GenericNode> LazyRender<N> {
    fn new(component: DynComponent<N>) -> Self {
        Self {
            component: Some(component),
            view: None,
        }
    }

    fn render(&mut self) -> View<N> {
        self.view
            .get_or_insert_with(|| self.component.take().unwrap().render())
            .clone()
    }
}
