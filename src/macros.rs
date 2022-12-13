#[macro_export]
#[doc(hidden)]
macro_rules! __view {
    (
        cx=($cx:expr)
        root=($root:ident)
        render={ $($render:tt)* }
        children={ $($children:tt)* }
        rest={}
    ) => {{
        let $root = $crate::create_component(
            $cx,
            move |root: $crate::element::$root<_>| { root $($render)*; },
        );
        $($children)*
        $root
    }
    };
    // Builder methods.
    (
        cx=($cx:expr)
        root=$root:tt
        render={ $($render:tt)* }
        children=$children:tt
        rest={ .$method:ident $args:tt $($rest:tt)* }
    ) => {
        $crate::__view! {
            cx=($cx)
            root=$root
            render={ $($render)* .$method $args }
            children=$children
            rest={ $($rest)* }
        }
    };
    (
        cx=($cx:expr)
        root=$root:tt
        render={ $($render:tt)* }
        children=$children:tt
        rest={ { $child:expr } $($rest:tt)* }
    ) => {
        $crate::__view! {
            cx=($cx)
            root=$root
            render={ $($render:tt)* .child($child) }
            children=$children
            rest={ $($rest)* }
        }
    };
    // Text nodes.
    (
        cx=($cx:expr)
        root=($root:ident)
        render=$render:tt
        children={ $($children:tt)* }
        rest={ $text:literal $($rest:tt)* }
    ) => {
        $crate::__view! {
            cx=($cx)
            root=($root)
            render=$render
            children={
                $($children)*
                let $root = $crate::Component::child($root,
                    $crate::view!($cx, text { .data($text) })
                );
            }
            rest={ $($rest)* }
        }
    };
    (
        cx=($cx:expr)
        root=($root:ident)
        render=$render:tt
        children={ $($children:tt)* }
        rest={ ($text:expr) $($rest:tt)* }
    ) => {
        $crate::__view! {
            cx=($cx)
            root=($root)
            render=$render
            children={
                $($children)*
                let $root = $crate::Component::child($root,
                    $crate::view!($cx, text { .data($text) })
                );
            }
            rest={ $($rest)* }
        }
    };
    // Child nodes.
    (
        cx=($cx:expr)
        root=($root:ident)
        render=$render:tt
        children={ $($children:tt)* }
        rest={ $child:ident $args:tt $($rest:tt)* }
    ) => {
        $crate::__view! {
            cx=($cx)
            root=($root)
            render=$render
            children={
                $($children)*
                let $root = $crate::Component::child($root,
                    $crate::view!($cx, $child $args)
                );
            }
            rest={ $($rest)* }
        }
    };
}

// TODO: use proc-macro to compile pre-defined templates
#[macro_export]
macro_rules! view {
    ($cx:expr, $root:ident { $($args:tt)* }) => {
        $crate::__view!(
            cx=($cx)
            root=($root)
            render={}
            children={}
            rest={ $($args)* }
        )
    };
}
