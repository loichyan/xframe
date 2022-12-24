use crate::Element;
use smallvec::SmallVec;
use xframe_core::{
    component::DynComponent, is_debug, view::ViewParentExt, GenericComponent, GenericElement,
    GenericNode, IntoReactive, Reactive, Template, Value, View,
};
use xframe_reactive::{untrack, Scope};

const INITIAL_BRANCH_SLOTS: usize = 2;

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
    children: SmallVec<[SwitchChild<N>; INITIAL_BRANCH_SLOTS]>,
}

pub struct SwitchChild<N> {
    cond: Reactive<bool>,
    content: DynComponent<N>,
}

struct Branch<N> {
    cond: Reactive<bool>,
    view: View<N>,
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
                    let branches = children
                        .into_iter()
                        .map(|SwitchChild { cond, content }| Branch {
                            cond,
                            // TODO: lazy rendering
                            view: content.render(),
                        })
                        // Fallback to a placeholder.
                        .chain(Some(Branch {
                            cond: Value(true),
                            view: placeholder,
                        }))
                        .collect::<SmallVec<[_; INITIAL_BRANCH_SLOTS]>>();
                    let mut current_index = branches.len() - 1;
                    move || {
                        for (index, branch) in branches.iter().enumerate() {
                            let Branch::<N> {
                                cond,
                                view: new_view,
                            } = branch;
                            // Only `cond` need to be tracked.
                            if cond.clone().into_value() {
                                untrack(|| {
                                    let current_view = dyn_view.get();
                                    // `dyn_view` may be moved or deleted, we need to
                                    // get its latest parent node.
                                    let parent = current_view.parent();
                                    if current_index != index {
                                        parent.replace_child(new_view, &current_view);
                                        current_index = index;
                                        debug_assert!(new_view.check_mount_order());
                                        dyn_view.set(new_view.clone());
                                    };
                                });
                                break;
                            }
                        }
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

impl<N> From<If<N>> for SwitchChild<N> {
    fn from(value: If<N>) -> Self {
        SwitchChild {
            cond: value.when.expect("`If::when` was not specified"),
            content: value.children.expect("`If::child` was not specified"),
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

impl<N> From<Else<N>> for SwitchChild<N> {
    fn from(value: Else<N>) -> Self {
        SwitchChild {
            cond: Value(true),
            content: value.children.expect("`Else::child` was not specified"),
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
