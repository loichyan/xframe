#[macro_export]
#[doc(hidden)]
macro_rules! __view {
    (cx=($cx:expr) output={ $($output:tt)* } rest={}) => {
        $($output)*
    };
    // Text nodes.
    (cx=($cx:expr) output={ $($output:tt)* } rest={ $text:literal $($rest:tt)* }) => {
        $crate::__view! {
            cx=($cx)
            output={
                $($output)*
                .child($crate::element::text($cx).data($text))
            }
            rest={ $($rest)* }
        }
    };
    (cx=($cx:expr) output={ $($output:tt)* } rest={ ($text:expr) $($rest:tt)* }) => {
        $crate::__view! {
            cx=($cx)
            output={
                $($output)*
                .child($crate::element::text($cx).data($text))
            }
            rest={ $($rest)* }
        }
    };
    // Builder methods.
    (cx=($cx:expr) output={ $($output:tt)* } rest={ .$method:ident $args:tt $($rest:tt)* }) => {
        $crate::__view! {
            cx=($cx)
            output={ $($output)* .$method $args }
            rest={ $($rest)* }
        }
    };
    // Child nodes.
    (cx=($cx:expr) output={ $($output:tt)* } rest={ { $child:expr } $($rest:tt)* }) => {
        $crate::__view! {
            cx=($cx)
            output={
                $($output)*
                .child($child)
            }
            rest={ $($rest)* }
        }
    };
    (cx=($cx:expr) output={ $($output:tt)* } rest={ $child:path { $($args:tt)* } $($rest:tt)* }) => {
        $crate::__view! {
            cx=($cx)
            output={
                $($output)*
                .child($crate::__view! {
                    cx=($cx)
                    output={ $child($cx) }
                    rest={ $($args)* }
                })
            }
            rest={ $($rest)* }
        }
    };
}

// TODO: use proc-macro to compile pre-defined templates
#[macro_export]
macro_rules! view {
    ($cx:expr, $root:path { $($args:tt)* }) => {
        $crate::__view!(
            cx=($cx)
            output={ $root($cx) }
            rest={ $($args)* }
        )
    };
}
