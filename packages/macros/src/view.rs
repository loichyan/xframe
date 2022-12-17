use crate::utils::QuoteSurround;
use proc_macro2::TokenStream;
use quote::{quote, ToTokens};
use syn::{
    braced, parenthesized,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    token, Expr, Ident, LitStr, Path, Result, Token,
};

new_type_quote! {
    XFRAME(::xframe);
    M_ELEMENT(::xframe::element);
    T_TEXT(#M_ELEMENT::text);
    VAR_ELEMENT(__element);
    VAR_CX(__cx);
    T_COMPONENT(__Component);
    T_GENERIC_COMPONENT(#XFRAME::GenericComponent);
    T_TEMPLATE(#XFRAME::Template);
    T_TEMPLATE_ID(#XFRAME::TemplateId);
    FN_CHILD(child);
    FN_BUILD(build);
    FN_VIEW(#XFRAME::view);
}

pub fn expand(input: ParseStream) -> Result<TokenStream> {
    Ok(input.parse::<View>()?.quote())
}

pub struct View {
    pub cx: Expr,
    pub comma_token: Token![,],
    pub root: ViewChild,
}

impl Parse for View {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(Self {
            cx: input.parse()?,
            comma_token: input.parse()?,
            root: input.parse()?,
        })
    }
}

impl View {
    pub fn quote(&self) -> TokenStream {
        let View { cx, root, .. } = self;
        let root = root.quote();
        quote!({
            struct #T_COMPONENT<F>(F);

            impl<N, F> #T_GENERIC_COMPONENT<N>
            for #T_COMPONENT<F>
            where
                N: #XFRAME::GenericNode,
                F: 'static + FnOnce() -> #T_TEMPLATE<N>,
            {
                fn id() -> Option<#T_TEMPLATE_ID> {
                    thread_local! {
                        static __ID: #T_TEMPLATE_ID = #T_TEMPLATE_ID::new();
                    }
                    Some(__ID.with(Clone::clone))
                }

                fn build_template(self) -> #T_TEMPLATE<N> {
                    (self.0)()
                }
            }

            #T_COMPONENT(move || {
                let #VAR_CX = #cx;
                #T_GENERIC_COMPONENT::build_template(#root)
            })
        })
    }
}

#[derive(Default)]
pub struct ViewArgs {
    pub props: Vec<ViewProp>,
    pub children: Vec<ViewChild>,
}

impl Parse for ViewArgs {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut new = Self::default();
        loop {
            if input.is_empty() {
                break;
            }
            if input.peek(Token![.]) {
                new.props.push(input.parse()?);
            } else {
                new.children.push(input.parse()?);
            }
        }
        Ok(new)
    }
}

impl ViewArgs {
    pub fn quote_props(&self) -> TokenStream {
        let props = self.props.iter().map(
            |ViewProp {
                 dot_token,
                 name,
                 paren_token,
                 args,
             }| {
                let args = QuoteSurround(paren_token, args);
                quote!(#dot_token #name #args)
            },
        );
        quote!(#(#props)*)
    }

    pub fn quote_children(&self) -> TokenStream {
        let children = self.children.iter().map(ViewChild::quote);
        let fn_ = FN_CHILD;
        quote!(#(.#fn_(#children))*)
    }
}

pub struct ViewProp {
    pub dot_token: Token![.],
    pub name: Ident,
    pub paren_token: token::Paren,
    pub args: Punctuated<Expr, Token![,]>,
}

impl Parse for ViewProp {
    fn parse(input: ParseStream) -> Result<Self> {
        let content;
        Ok(Self {
            dot_token: input.parse()?,
            name: input.parse()?,
            paren_token: parenthesized!(content in input),
            args: content.parse_terminated(Expr::parse)?,
        })
    }
}

pub enum ViewChild {
    Literal(LitStr),
    Text {
        paren_token: token::Paren,
        value: Expr,
    },
    Expr {
        brace_token: token::Brace,
        value: Expr,
    },
    View(ViewComponent),
}

impl Parse for ViewChild {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(if input.peek(LitStr) {
            Self::Literal(input.parse()?)
        } else if input.peek(token::Paren) {
            let content;
            Self::Text {
                paren_token: parenthesized!(content in input),
                value: content.parse()?,
            }
        } else if input.peek(token::Brace) {
            let content;
            Self::Expr {
                brace_token: braced!(content in input),
                value: content.parse()?,
            }
        } else {
            Self::View(input.parse()?)
        })
    }
}

impl ViewChild {
    pub fn quote(&self) -> TokenStream {
        match self {
            Self::Literal(lit) => {
                quote!(#FN_VIEW(
                        #VAR_CX,
                        move |#VAR_ELEMENT: #T_TEXT::<_>| { #VAR_ELEMENT.data(#lit) },
                ))
            }
            Self::Text { paren_token, value } => {
                let value = QuoteSurround(paren_token, value);
                quote!(#FN_VIEW(
                    #VAR_CX,
                    move |#VAR_ELEMENT: #T_TEXT::<_>| { #VAR_ELEMENT.data #value },
                ))
            }
            Self::Expr { value, .. } => value.to_token_stream(),
            Self::View(view) => view.quote(),
        }
    }
}

pub struct ViewComponent {
    pub path: Path,
    pub brace_token: token::Brace,
    pub args: ViewArgs,
}

impl Parse for ViewComponent {
    fn parse(input: ParseStream) -> Result<Self> {
        let content;
        Ok(Self {
            path: input.parse()?,
            brace_token: braced!(content in input),
            args: content.parse()?,
        })
    }
}

impl ViewComponent {
    pub fn quote(&self) -> TokenStream {
        let Self { path, args, .. } = self;
        let is_builtin = if let Some(ident) = path.get_ident() {
            ident
                .to_string()
                .trim_end_matches('_')
                .chars()
                .all(|c| c.is_ascii_lowercase())
        } else {
            false
        };
        let props = args.quote_props();
        let children = args.quote_children();
        if is_builtin {
            quote!({
                #FN_VIEW(
                    #VAR_CX,
                    move |#VAR_ELEMENT: #M_ELEMENT::#path::<_>| { #VAR_ELEMENT #props },
                )
                #children
            })
        } else {
            quote!({
                #path::<_>(#VAR_CX)
                #props
                #children
                .#FN_BUILD()
            })
        }
    }
}
