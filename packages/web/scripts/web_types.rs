use heck::{ToKebabCase, ToPascalCase};
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
new_type_quote!(ELEMENT_TYPES(super::element_types));
new_type_quote!(EVENT_TYPES(super::event_types));
new_type_quote!(ATTR_TYPES(super::attr_types));

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
                    .map(|lit| Variant {
                        name: lit.to_pascal_case().to_ident(),
                        literal: lit.to_lit_str(),
                    })
                    .collect();
                attr_types.insert(attr.ty.clone(), AttrLitType { variants });
            }
        }
        for event in element.events.iter() {
            if event_types.contains_key(&event.ty) {
                continue;
            }
            let unstable = matches!(event.original.web_sys_type, "ClipboardEvent");
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
        #[cfg(feature = "extra-elements")]
        pub mod output {
            use super::input;
            pub mod elements { #(#element_fns)* }
            pub mod element_types { #(#element_structs)* #(#element_types)* }
            #[cfg(feature = "extra-attributes")]
            pub mod attr_types { #(#attr_types)* }
            #[cfg(feature = "extra-events")]
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
            web_sys_type,
            events,
            attributes,
        } = input;
        Self {
            key: name.to_kebab_case().to_lit_str(),
            ty: web_sys_type.to_ident(),
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
            pub fn #fn_<N: #INPUT::GenericNode>() -> #ELEMENT_TYPES::#struct_<N> {
                #ELEMENT_TYPES::#struct_::<N>(#INPUT::BaseElement::create(#key))
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
        quote!(
            pub struct #struct_<N>(pub(crate) #INPUT::BaseElement<N>);

            const _: () = {
                use #INPUT::{GenericElement, GenericNode, IntoAttribute};
                use std::borrow::Cow;

                impl<N: GenericNode> GenericElement
                for #struct_<N>
                {
                    type Node = N;
                    fn into_node(self) -> Self::Node {
                        GenericElement::into_node(self.0)
                    }
                }

                impl<N> AsRef<#ELEMENT_TYPES::#ty> for #struct_<N>
                where
                    N: GenericNode + AsRef<#WEB_SYS::Node>,
                {
                    fn as_ref(&self) -> &#ELEMENT_TYPES::#ty {
                        self.0.as_web_sys_element()
                    }
                }

                #[cfg(feature = "extra-attributes")]
                impl<N: GenericNode> #struct_<N> { #(#attr_fns)* }

                #[cfg(feature = "extra-events")]
                impl<N: GenericNode> #struct_<N> { #(#event_fns)* }

                impl<N: GenericNode> #struct_<N> {
                    pub fn attribute<K: Into<Cow<'static, str>>, V: IntoAttribute>(
                        self,
                        name: K,
                        val: V,
                    ) -> Self {
                        self.0.node().set_attribute(name.into(), val.into_attribute());
                        self
                    }

                    /// Add a class to this element.
                    pub fn class<T: Into<Cow<'static, str>>>(self, name: T) -> Self {
                        self.0.node().add_class(name.into());
                        self
                    }

                    pub fn child<E>(self, element: E) -> Self
                    where
                        E: GenericElement<Node = N>,
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
                }
            };
        )
    }
}

struct Attribute<'a> {
    original: &'a web_types::Attribute<'a>,
    key: LitStr,
    generic: Option<TokenStream>,
    ty: Ident,
    fn_: Ident,
}

impl<'a> Attribute<'a> {
    fn from_web(input: &'a web_types::Attribute<'a>) -> Self {
        let web_types::Attribute { name, js_type } = input;
        let mut generic = None;
        let ty = match js_type {
            JsType::Type(ty) => match *ty {
                "string" => {
                    generic = Some(quote!(T: #INPUT::IntoAttribute,));
                    "T".to_ident()
                }
                "number" => "i32".to_ident(),
                "boolean" => "bool".to_ident(),
                _ => panic!("unknown js type '{ty}'"),
            },
            JsType::Literals(lits) => lits.name.to_ident(),
        };
        Self {
            original: input,
            key: name.to_kebab_case().to_lit_str(),
            generic,
            ty,
            fn_: name.to_ident(),
        }
    }

    fn quote_fn(&self) -> TokenStream {
        let Self {
            original,
            key,
            generic,
            ty,
            fn_,
        } = self;
        let path = if let JsType::Literals(_) = original.js_type {
            Some(quote!(#ATTR_TYPES::))
        } else {
            None
        };
        quote!(
            pub fn #fn_<#generic>(self, val: #path #ty) -> Self {
                self.0.set_attribute(#key, val);
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
        let web_types::Event { name, web_sys_type } = input;
        Self {
            original: input,
            key: name.to_kebab_case().to_lit_str(),
            ty: web_sys_type.to_ident(),
            fn_: format_ident!("on_{}", name.to_ident()),
        }
    }

    fn quote_fn(&self) -> TokenStream {
        let Self { key, ty, fn_, .. } = self;
        quote!(
            pub fn #fn_(
                self,
                handler: impl #INPUT::EventHandler<#EVENT_TYPES::#ty>,
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
        let arms = ty
            .variants
            .iter()
            .map(|Variant { name, literal }| quote!(Self::#name => #literal));
        quote!(
            pub enum #name { #(#variants,)* }

            impl #INPUT::IntoAttribute for #name {
                fn into_attribute(self) -> #INPUT::Attribute {
                    #INPUT::Attribute::from_literal(match self { #(#arms,)* })
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