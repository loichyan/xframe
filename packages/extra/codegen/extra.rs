use heck::{ToKebabCase, ToPascalCase, ToSnakeCase};
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote, ToTokens};
use std::collections::BTreeMap;
use syn::{Ident, LitStr};
use web_types::JsType;

macro_rules! new_type_quote {
    ($($name:ident $tt:tt;)*) => {
        $(new_type_quote!($name $tt);)*
    };
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

new_type_quote! {
    M_CORE(#M_INPUT::core);
    M_INPUT(super::input);
    M_REACTIVE(#M_INPUT::reactive);
    M_WEB(#M_INPUT::web);
    M_WEB_SYS(#M_INPUT::web_sys);

    M_ATTR_TYPES(super::attr_types);
    M_ELEMENT_TYPES(super::element_types);
    M_EVENT_TYPES(super::event_types);

    T_WEB_NODE(#M_WEB::WebNode);
    T_GENERIC_COMPONENT(#M_CORE::GenericComponent);
    T_GENERIC_ELEMENT(#M_WEB::GenericElement);
    T_GENERIC_NODE(#M_CORE::GenericNode);
    T_INTO_REACTIVE(#M_CORE::IntoReactive);
    T_INTO_EVENT_HANDLER(#M_CORE::IntoEventHandler);

    BASE_ELEMENT(#M_INPUT::BaseElement);
    COW_STR(::std::borrow::Cow::<'static, str>);
    ELEMENT(#M_CORE::component::Element);
    NODE_TYPE(#M_CORE::NodeType);
    OUTPUT(#M_CORE::RenderOutput);
    REACTIVE(#M_CORE::Reactive);
    SCOPE(#M_REACTIVE::Scope);
    STRING_LIKE(#M_CORE::StringLike);
}

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
    let mut element_types = BTreeMap::default();
    for element in input {
        let element = Element::from_web(element);
        if !element_types.contains_key(&element.ty) {
            element_types.insert(element.ty.clone(), element.js_ty.clone());
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
            event_types.insert(
                event.ty.clone(),
                EventType {
                    unstable,
                    js_ty: event.js_ty.clone(),
                },
            );
        }
        elements.push(element);
    }
    let attr_types = attr_types.iter().map(QuoteAttrType);
    let event_types = event_types.iter().map(|(k, v)| QuoteEventType((k, v)));
    let element_definitions = elements.iter().map(Element::quote);
    let element_types = element_types.iter().map(QuoteElementType);
    quote!(
        #[cfg(feature = "elements")]
        #[allow(clippy::all)]
        pub mod output {
            use super::input;
            pub mod elements { #(#element_definitions)* }
            pub mod element_types { #(#element_types)* }
            #[cfg(feature = "attributes")]
            pub mod attr_types {
                pub(super) type JsBoolean = #M_INPUT::JsBoolean;
                pub(super) type JsNumber = #M_INPUT::JsNumber;
                pub(super) type JsString = #M_INPUT::JsString;
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
    js_ty: Ident,
    fn_: Ident,
    attributes: Vec<Property<'a>>,
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
        let js_ty = if let Some(js_class) = js_class.strip_suffix("Element") {
            if let Some(js_class) = js_class.strip_prefix("HTML") {
                let js_class = match js_class {
                    "BR" => "Br",
                    "HR" => "Hr",
                    "LI" => "Li",
                    _ => js_class,
                };
                format_ident!("Html{js_class}Element")
            } else if let Some(js_class) = js_class.strip_prefix("SVG") {
                if let Some(js_class) = js_class.strip_prefix("FE") {
                    format_ident!("Svgfe{js_class}Element")
                } else {
                    let js_class = match js_class {
                        "G" => "g",
                        "SVG" => "svg",
                        "MPath" => "mPath",
                        "TSpan" => "tSpan",
                        _ => js_class,
                    };
                    format_ident!("Svg{js_class}Element")
                }
            } else {
                panic!("unknown js class: {js_class}");
            }
        } else {
            panic!("unknown js class: {js_class}");
        };
        let mut fn_ = name.to_snake_case();
        if matches!(&*fn_, "use") {
            fn_.push('_');
        }
        Self {
            key: name.to_kebab_case().to_lit_str(),
            ty: name.to_pascal_case().to_ident(),
            js_ty,
            fn_: fn_.to_ident(),
            attributes: attributes
                .iter()
                .map(|attr| Property::from_web(input, attr))
                .collect(),
            events: events.iter().map(Event::from_web).collect(),
        }
    }

    fn quote(&self) -> TokenStream {
        let Self {
            ty,
            fn_,
            attributes,
            events,
            key,
            ..
        } = self;
        let attr_fns = attributes.iter().map(Property::quote_fn);
        let event_fns = events.iter().map(Event::quote_fn);
        quote!(
            #[allow(non_camel_case_types)]
            pub struct #fn_<N> {
                inner: #BASE_ELEMENT<N>
            }

            pub fn #fn_<N: #T_GENERIC_NODE>(cx: #SCOPE) -> #fn_<N> {
                #fn_ {
                    inner: #BASE_ELEMENT::new::<#fn_<N>>(cx),
                }
            }

            impl<N: #T_GENERIC_NODE> AsRef<#ELEMENT<N>> for #fn_<N> {
                fn as_ref(&self) -> &#ELEMENT<N> {
                    self.inner.as_element()
                }
            }

            impl<N: #T_GENERIC_NODE> AsMut<#ELEMENT<N>> for #fn_<N> {
                fn as_mut(&mut self) -> &mut #ELEMENT<N> {
                    self.inner.as_element_mut()
                }
            }

            impl<N: #T_GENERIC_NODE> #T_GENERIC_COMPONENT<N>
            for #fn_<N>
            {
                fn render(self) -> #OUTPUT<N> {
                    self.inner.render()
                }
            }

            impl<N: #T_GENERIC_NODE> #T_GENERIC_ELEMENT<N> for #fn_<N> {
                const TYPE: #NODE_TYPE = #NODE_TYPE::Tag(#COW_STR::Borrowed(#key));
            }

            impl<N> AsRef<#M_ELEMENT_TYPES::#ty> for #fn_<N>
            where
                N: #T_GENERIC_NODE + AsRef<#M_WEB_SYS::Node>,
            {
                fn as_ref(&self) -> &#M_ELEMENT_TYPES::#ty {
                    self.inner.as_web_sys_element()
                }
            }

            #[cfg(feature = "attributes")]
            impl<N: #T_GENERIC_NODE> #fn_<N> { #(#attr_fns)* }

            #[cfg(feature = "events")]
            impl<N: #T_WEB_NODE> #fn_<N> { #(#event_fns)* }
        )
    }
}

struct Property<'a> {
    original: &'a web_types::Property<'a>,
    key: LitStr,
    ty: Ident,
    fn_: Ident,
    attribute: bool,
}

impl<'a> Property<'a> {
    fn from_web(ele: &web_types::Element, input: &'a web_types::Property<'a>) -> Self {
        let web_types::Property {
            name,
            js_type,
            attribute,
        } = input;
        let ty = match js_type {
            JsType::Type(ty) => match *ty {
                "string" => "JsString".to_ident(),
                "number" => "JsNumber".to_ident(),
                "boolean" => "JsBoolean".to_ident(),
                _ => panic!("unknown js type '{ty}'"),
            },
            JsType::Literals(_) => format!("{}-{}", ele.name, name).to_pascal_case().to_ident(),
        };
        let mut fn_ = name.to_snake_case();
        if matches!(*name, "as" | "in" | "async" | "loop" | "type") {
            fn_.push('_');
        }
        Self {
            original: input,
            key: name.to_lit_str(),
            ty,
            fn_: fn_.to_ident(),
            attribute: *attribute,
        }
    }

    fn quote_fn(&self) -> TokenStream {
        let Self {
            key,
            ty,
            fn_,
            attribute,
            ..
        } = self;
        let f = if *attribute {
            quote!(set_attribute_literal)
        } else {
            quote!(set_property_literal)
        };
        quote!(
            pub fn #fn_<T: #T_INTO_REACTIVE<#M_ATTR_TYPES::#ty>>(self, val: T) -> Self {
                self.inner.#f(#key, val);
                self
            }
        )
    }
}

struct Event<'a> {
    original: &'a web_types::Event<'a>,
    key: LitStr,
    ty: Ident,
    js_ty: Ident,
    fn_: Ident,
}

impl<'a> Event<'a> {
    fn from_web(input: &'a web_types::Event<'a>) -> Self {
        let web_types::Event { name, js_class } = input;
        let js_ty = match *js_class {
            "UIEvent" => "UiEvent".to_ident(),
            "FormDataEvent" => "Event".to_ident(),
            _ => js_class.to_ident(),
        };
        Self {
            original: input,
            key: name.to_kebab_case().to_lit_str(),
            ty: name.to_pascal_case().to_ident(),
            js_ty,
            fn_: format_ident!("on_{}", name.to_ident()),
        }
    }

    fn quote_fn(&self) -> TokenStream {
        let Self { key, ty, fn_, .. } = self;
        quote!(
            pub fn #fn_(
                self,
                handler: impl #T_INTO_EVENT_HANDLER<#M_EVENT_TYPES::#ty>,
            ) -> Self {
                self.inner.listen_event(#key, handler);
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
    js_ty: Ident,
}

struct QuoteEventType<'a>((&'a Ident, &'a EventType));

impl ToTokens for QuoteEventType<'_> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let Self((name, EventType { unstable, js_ty })) = self;
        if *unstable {
            quote!(
                #[cfg(web_sys_unstable_apis)]
                pub type #name = #M_WEB_SYS::#js_ty;
                #[cfg(not(web_sys_unstable_apis))]
                pub type #name = #M_WEB_SYS::Event;
            )
        } else {
            quote!(
                pub type #name = #M_WEB_SYS::#js_ty;
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

            impl From<#name> for #STRING_LIKE {
                fn from(t: #name) -> Self {
                    #STRING_LIKE::Literal(t.into())
                }
            }

            impl From<#name> for #REACTIVE<#name> {
                fn from(t: #name) -> Self {
                    #REACTIVE::Variable(t)
                }
            }

            impl From<#name> for #REACTIVE<#STRING_LIKE> {
                fn from(t: #name) -> Self {
                    #REACTIVE::Variable(t.into())
                }
            }
        )
        .to_tokens(tokens);
    }
}

struct QuoteElementType<'a>((&'a Ident, &'a Ident));

impl ToTokens for QuoteElementType<'_> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let Self((ty, js_ty)) = self;
        quote!(
            // TODO: suffix with `Element`
            pub type #ty = #M_WEB_SYS::#js_ty;
        )
        .to_tokens(tokens);
    }
}
