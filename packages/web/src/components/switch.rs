use crate::Element;
use smallvec::SmallVec;
use xframe_core::{
    component::DynComponent, is_dev, GenericComponent, GenericElement, GenericNode, IntoReactive,
    Reactive, Template, Value, View,
};
use xframe_reactive::{untrack, Scope};

const INITIAL_BRANCH_SLOTS: usize = 4;

define_placeholder!(Placeholder("PLACEHOLDER FOR `xframe::Switch` COMPONENT"));

#[allow(non_snake_case)]
pub fn Switch<N: GenericNode>(cx: Scope) -> Switch<N> {
    Switch {
        cx,
        children: Default::default(),
    }
}

pub struct Switch<N: GenericNode> {
    cx: Scope,
    children: SmallVec<[SwitchChild<N>; INITIAL_BRANCH_SLOTS]>,
}

pub struct SwitchChild<N: GenericNode> {
    cond: Reactive<bool>,
    content: DynComponent<N>,
}

struct Branch<N: GenericNode> {
    cond: Reactive<bool>,
    view: View<N>,
}

impl<N: GenericNode> GenericComponent<N> for Switch<N> {
    fn build_template(self) -> Template<N> {
        let Self { cx, children } = self;
        Element(cx)
            .with_view(move |placeholder: Placeholder<N>| {
                let placeholder = placeholder.into_node();
                let branches = children
                    .into_iter()
                    .map(|SwitchChild { cond, content }| Branch {
                        cond,
                        view: untrack(|| content.render()),
                    })
                    // Add a default branch.
                    .chain(Some(Branch {
                        cond: Value(true),
                        view: View::node(placeholder.clone()),
                    }))
                    .collect::<SmallVec<[_; INITIAL_BRANCH_SLOTS]>>();
                let dyn_view = View::dyn_(cx, View::node(placeholder));
                cx.create_effect({
                    let dyn_view = dyn_view.clone();
                    move || {
                        for branch in branches.iter() {
                            let Branch::<N> {
                                cond,
                                view: new_view,
                            } = branch;
                            if cond.clone().into_value() {
                                untrack(|| {
                                    let current_view = dyn_view.get();
                                    let current_first = current_view.first();
                                    let parent = current_first.parent().unwrap();
                                    let new_first = new_view.first();
                                    if current_first.ne(&new_first) {
                                        current_view.replace_with(&parent, new_view);
                                        dyn_view.set(new_view.clone());
                                    }
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

pub struct If<N: GenericNode> {
    cx: Scope,
    when: Option<Reactive<bool>>,
    children: Option<DynComponent<N>>,
}

impl<N: GenericNode> From<If<N>> for SwitchChild<N> {
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
            panic!("`If::when` has already be specified");
        }
        self.when = Some(when.into_reactive());
        self
    }

    pub fn child<C: GenericComponent<N>>(mut self, child: C) -> If<N> {
        if self.children.is_some() {
            panic!("`If::child` has already be specified");
        }
        self.children = Some(child.into_dyn_component());
        self
    }
}

#[allow(non_snake_case)]
pub fn Else<N: GenericNode>(cx: Scope) -> Else<N> {
    Else { cx, children: None }
}

pub struct Else<N: GenericNode> {
    cx: Scope,
    children: Option<DynComponent<N>>,
}

impl<N: GenericNode> From<Else<N>> for SwitchChild<N> {
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
        if is_dev!() {
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
            panic!("`Else::child` has already be specified");
        }
        self.children = Some(child.into_dyn_component());
        self
    }
}
