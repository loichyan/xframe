use heck::{ToKebabCase, ToPascalCase, ToSnakeCase};
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote, ToTokens};
use std::collections::{BTreeMap, BTreeSet};
use syn::{Ident, LitStr};
use web_types::JsType;

macro_rules! new_type_quote {
    ($name:ident($($tt:tt)*)) => {
        #[allow(non_camel_case_types)]
        #[allow(clippy::upper_case_acronyms)]
        struct $name;
        impl ::quote::ToTokens for $name {
            fn to_tokens(&self, tokens: &mut ::proc_macro2::TokenStream) {
                ::quote::quote!($($tt)*).to_tokens(tokens);
            }
        }
    };
}

new_type_quote!(INPUT(super::input));
new_type_quote!(WEB_SYS(#INPUT::web_sys));
new_type_quote!(XFRAME(#INPUT::xframe));
new_type_quote!(GENERIC_NODE(#INPUT::core::GenericNode));
new_type_quote!(GENERIC_ELEMENT(#INPUT::core::GenericElement));
new_type_quote!(ATTRIBUTE(#INPUT::core::Attribute));
new_type_quote!(REACTIVE(#INPUT::core::Reactive));
new_type_quote!(INTO_REACTIVE(#INPUT::core::IntoReactive));
new_type_quote!(INTO_EVENT_HANDLER(#INPUT::core::IntoEventHandler));
new_type_quote!(COW_STR(::std::borrow::Cow<'static, str>));
new_type_quote!(ELEMENT_TYPES(super::element_types));
new_type_quote!(ATTR_TYPES(super::attr_types));
new_type_quote!(EVENT_TYPES(super::event_types));

trait StrExt: AsRef<str> {
    fn to_lit_str(&self) -> LitStr {
        LitStr::new(self.as_ref(), Span::call_site())
    }

    fn to_ident(&self) -> Ident {
        Ident::new(self.as_ref(), Span::call_site())
    }
}

impl<T: AsRef<str>> StrExt for T {}

pub fn expand(input: &[web_types::Element]) -> TokenStream {
    let mut elements = Vec::default();
    let mut attr_types = BTreeMap::default();
    let mut event_types = BTreeMap::default();
    let mut element_types = BTreeSet::default();
    for element in input {
        let element = Element::from_web(element);
        if !element_types.contains(&element.ty) {
            element_types.insert(element.ty.clone());
        }
        for attr in element.attributes.iter() {
            if let JsType::Literals(lits) = &attr.original.js_type {
                if attr_types.contains_key(&attr.ty) {
                    continue;
                }
                let variants = lits
                    .values
                    .iter()
                    .copied()
                    .flat_map(Variant::from_web)
                    .collect();
                attr_types.insert(attr.ty.clone(), AttrLitType { variants });
            }
        }
        for event in element.events.iter() {
            if event_types.contains_key(&event.ty) {
                continue;
            }
            let unstable = matches!(event.original.js_class, "ClipboardEvent");
            event_types.insert(event.ty.clone(), EventType { unstable });
        }
        elements.push(element);
    }
    let element_types = element_types.iter().map(QuoteElementType);
    let attr_types = attr_types.iter().map(QuoteAttrType);
    let event_types = event_types.iter().map(QuoteEventType);
    let element_fns = elements.iter().map(Element::quote_fn);
    let element_structs = elements.iter().map(Element::quote_struct);
    quote!(
        #[cfg(feature = "elements")]
        #[allow(clippy::all)]
        pub mod output {
            use super::input;
            pub mod elements { #(#element_fns)* }
            pub mod element_types { #(#element_structs)* #(#element_types)* }
            #[cfg(feature = "attributes")]
            pub mod attr_types {
                pub(super) type JsBoolean = #INPUT::JsBoolean;
                pub(super) type JsNumber = #INPUT::JsNumber;
                pub(super) type JsString = #INPUT::JsString;
                #(#attr_types)*
            }
            #[cfg(feature = "events")]
            pub mod event_types { #(#event_types)* }
        }
    )
}

struct Element<'a> {
    key: LitStr,
    ty: Ident,
    fn_: Ident,
    struct_: Ident,
    attributes: Vec<Attribute<'a>>,
    events: Vec<Event<'a>>,
}

impl<'a> Element<'a> {
    fn from_web(input: &'a web_types::Element<'a>) -> Self {
        let web_types::Element {
            name,
            js_class,
            events,
            attributes,
        } = input;
        let ty = match *js_class {
            "HTMLBRElement" => "HtmlBrElement".to_ident(),
            "HTMLHRElement" => "HtmlHrElement".to_ident(),
            "HTMLLIElement" => "HtmlLiElement".to_ident(),
            _ => format!("Html{}", &js_class[4..]).to_ident(),
        };
        Self {
            key: name.to_kebab_case().to_lit_str(),
            ty,
            fn_: name.to_ident(),
            struct_: name.to_pascal_case().to_ident(),
            attributes: attributes
                .iter()
                .filter_map(|attr| {
                    if attr.name == "class" {
                        None
                    } else {
                        Some(Attribute::from_web(attr))
                    }
                })
                .collect(),
            events: events.iter().map(Event::from_web).collect(),
        }
    }

    fn quote_fn(&self) -> TokenStream {
        let Self {
            key, fn_, struct_, ..
        } = self;
        quote!(
            pub fn #fn_<N: #GENERIC_NODE>(cx: #XFRAME::Scope) -> #ELEMENT_TYPES::#struct_<N> {
                #ELEMENT_TYPES::#struct_::<N>(#INPUT::BaseElement::create(#key, cx))
            }
        )
    }

    fn quote_struct(&self) -> TokenStream {
        let Self {
            ty,
            struct_,
            attributes,
            events,
            ..
        } = self;
        let attr_fns = attributes.iter().map(Attribute::quote_fn);
        let event_fns = events.iter().map(Event::quote_fn);
        let default_methods = self.quote_default_methods();
        quote!(
            pub struct #struct_<N>(pub(crate) #INPUT::BaseElement<N>);

            impl<N: #GENERIC_NODE> #GENERIC_ELEMENT
            for #struct_<N>
            {
                type Node = N;
                fn into_node(self) -> Self::Node {
                    #GENERIC_ELEMENT::into_node(self.0)
                }
            }

            impl<N> AsRef<#ELEMENT_TYPES::#ty> for #struct_<N>
            where
                N: #GENERIC_NODE + AsRef<#WEB_SYS::Node>,
            {
                fn as_ref(&self) -> &#ELEMENT_TYPES::#ty {
                    self.0.as_web_sys_element()
                }
            }

            impl<N: #GENERIC_NODE> #struct_<N> { #default_methods }

            #[cfg(feature = "attributes")]
            impl<N: #GENERIC_NODE> #struct_<N> { #(#attr_fns)* }

            #[cfg(feature = "events")]
            impl<N: #GENERIC_NODE<Event = #WEB_SYS::Event>> #struct_<N> { #(#event_fns)* }
        )
    }

    fn quote_default_methods(&self) -> TokenStream {
        quote!(
            pub fn attr<K: Into<#COW_STR>, V: #INTO_REACTIVE<#ATTRIBUTE>>(
                self,
                name: K,
                val: V,
            ) -> Self {
                self.0.set_property(name.into(), val);
                self
            }

            /// Add a class to this element.
            pub fn class<T: Into<#COW_STR>>(self, name: T) -> Self {
                self.0.node().add_class(name.into());
                self
            }

            pub fn child<E>(self, element: E) -> Self
            where
                E: #GENERIC_ELEMENT<Node = N>,
            {
                self.0.node().append_child(element.into_node());
                self
            }

            pub fn children<I>(self, nodes: I) -> Self
            where
                I: IntoIterator<Item = N>,
            {
                let node = self.0.node();
                for child in nodes {
                    node.append_child(child);
                }
                self
            }
        )
    }
}

struct Attribute<'a> {
    original: &'a web_types::Attribute<'a>,
    key: LitStr,
    ty: Ident,
    fn_: Ident,
}

impl<'a> Attribute<'a> {
    fn from_web(input: &'a web_types::Attribute<'a>) -> Self {
        let web_types::Attribute { name, js_type } = input;
        let ty = match js_type {
            JsType::Type(ty) => match *ty {
                "string" => "JsString".to_ident(),
                "number" => "JsNumber".to_ident(),
                "boolean" => "JsBoolean".to_ident(),
                _ => panic!("unknown js type '{ty}'"),
            },
            JsType::Literals(lits) => lits.name.to_ident(),
        };
        let mut fn_ = name.to_snake_case();
        if matches!(*name, "type" | "loop" | "async" | "as") {
            fn_.push('_');
        }
        Self {
            original: input,
            key: name.to_lit_str(),
            ty,
            fn_: fn_.to_ident(),
        }
    }

    fn quote_fn(&self) -> TokenStream {
        let Self { key, ty, fn_, .. } = self;
        quote!(
            pub fn #fn_<T: #INTO_REACTIVE<#ATTR_TYPES::#ty>>(self, val: T) -> Self {
                self.0.set_property_literal(#key, val);
                self
            }
        )
    }
}

struct Event<'a> {
    original: &'a web_types::Event<'a>,
    key: LitStr,
    ty: Ident,
    fn_: Ident,
}

impl<'a> Event<'a> {
    fn from_web(input: &'a web_types::Event<'a>) -> Self {
        let web_types::Event { name, js_class } = input;
        let ty = match *js_class {
            "UIEvent" => "UiEvent".to_ident(),
            "FormDataEvent" => "Event".to_ident(),
            _ => js_class.to_ident(),
        };
        Self {
            original: input,
            key: name.to_kebab_case().to_lit_str(),
            ty,
            fn_: format_ident!("on_{}", name.to_ident()),
        }
    }

    fn quote_fn(&self) -> TokenStream {
        let Self { key, ty, fn_, .. } = self;
        quote!(
            pub fn #fn_(
                self,
                handler: impl #INTO_EVENT_HANDLER<#EVENT_TYPES::#ty>,
            ) -> Self {
                self.0.listen_event(#key, handler);
                self
            }
        )
    }
}

struct AttrLitType {
    variants: Vec<Variant>,
}

struct Variant {
    name: Ident,
    literal: LitStr,
}

impl Variant {
    fn from_web(lit: &str) -> Option<Self> {
        if lit.is_empty() {
            return None;
        }
        Some(Self {
            name: lit.to_pascal_case().to_ident(),
            literal: lit.to_lit_str(),
        })
    }
}

struct EventType {
    unstable: bool,
}

struct QuoteEventType<'a>((&'a Ident, &'a EventType));

impl ToTokens for QuoteEventType<'_> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let Self((name, ty)) = self;
        if ty.unstable {
            quote!(
                #[cfg(web_sys_unstable_apis)]
                pub type #name = #WEB_SYS::#name;
                #[cfg(not(web_sys_unstable_apis))]
                pub type #name = #WEB_SYS::Event;
            )
        } else {
            quote!(
                pub type #name = #WEB_SYS::#name;
            )
        }
        .to_tokens(tokens);
    }
}

struct QuoteAttrType<'a>((&'a Ident, &'a AttrLitType));

impl ToTokens for QuoteAttrType<'_> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let Self((name, ty)) = self;
        let variants = ty.variants.iter().map(|v| {
            let Variant { name, literal } = v;
            quote!(
                #[doc = "Literal of `"]
                #[doc = #literal]
                #[doc = "`."]
                #name
            )
        });
        let arms = ty.variants.iter().map(
            |Variant {
                 name: variant,
                 literal,
             }| quote!(#name::#variant => #literal),
        );
        quote!(
            pub enum #name { #(#variants,)* }

            impl From<#name> for &'static str {
                fn from(t: #name) -> Self {
                    match t { #(#arms,)* }
                }
            }

            impl From<#name> for #ATTRIBUTE {
                fn from(t: #name) -> Self {
                    #ATTRIBUTE::from_literal(t.into())
                }
            }

            impl From<#name> for #REACTIVE<#name> {
                fn from(t: #name) -> Self {
                    #REACTIVE::Value(t)
                }
            }

            impl From<#name> for #REACTIVE<#ATTRIBUTE> {
                fn from(t: #name) -> Self {
                    #REACTIVE::Value(t.into())
                }
            }
        )
        .to_tokens(tokens);
    }
}

struct QuoteElementType<'a>(&'a Ident);

impl ToTokens for QuoteElementType<'_> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let Self(name) = self;
        quote!(
            pub type #name = #WEB_SYS::#name;
        )
        .to_tokens(tokens);
    }
}
