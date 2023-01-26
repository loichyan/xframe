use crate::utils::QuoteSurround;
use proc_macro2::TokenStream;
use quote::{quote, ToTokens};
use syn::{
    braced, bracketed, parenthesized,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    token, Expr, Ident, LitStr, Path, Result, Token,
};

new_type_quote! {
    XFRAME(::xframe);
    RT(#XFRAME::__private);
    M_ELEMENTS(#XFRAME::elements);
    VAR_CX(__cx);
    VAR_VIEW(__view);
    T_COMPONENT(__Component);
    T_GENERIC_COMPONENT(#XFRAME::GenericComponent);
    T_TEMPLATE(#XFRAME::Template);
    T_TEMPLATE_ID(#XFRAME::TemplateId);
    T_TEXT(#M_ELEMENTS::text);
    FN_CHILD(child);
    FN_VIEW_ELEMENT(#RT::view_element);
    FN_VIEW_TEXT(#RT::view_text);
    FN_VIEW_TEXT_LITERAL(#RT::view_text_literal);
    FN_VIEW_COMPONENT(#RT::view_component);
    FN_VIEW_FRAGMENT(#RT::view_fragment);
    FN_VIEW(#XFRAME::view);
}

// TODO: better spanned
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
            let #VAR_CX = #cx;
            #RT::view_root(
                #VAR_CX,
                #XFRAME::id!(),
                #root,
            )
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
        quote_children(&self.children)
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
    Element(ViewElement),
    Fragment {
        bracket_token: token::Bracket,
        children: Vec<ViewChild>,
    },
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
        } else if input.peek(token::Bracket) {
            let content;
            Self::Fragment {
                bracket_token: bracketed!(content in input ),
                children: {
                    let mut children = Vec::new();
                    loop {
                        if content.is_empty() {
                            break;
                        }
                        children.push(content.parse()?);
                    }
                    children
                },
            }
        } else {
            Self::Element(input.parse()?)
        })
    }
}

impl ViewChild {
    pub fn quote(&self) -> TokenStream {
        match self {
            Self::Literal(lit) => quote!({ #FN_VIEW_TEXT_LITERAL(#VAR_CX, #lit) }),
            Self::Text { value, .. } => quote!({ #FN_VIEW_TEXT(#VAR_CX, #value) }),
            Self::Expr { value, .. } => value.to_token_stream(),
            Self::Element(view) => view.quote(),
            Self::Fragment { children, .. } => {
                // TODO: always use fragment
                let children = quote_children(children);
                quote!({ #FN_VIEW_FRAGMENT(
                    #VAR_CX,
                    move |#VAR_VIEW| { #VAR_VIEW #children },
                ) })
            }
        }
    }
}

pub struct ViewElement {
    pub path: Path,
    pub brace_token: token::Brace,
    pub args: ViewArgs,
}

impl Parse for ViewElement {
    fn parse(input: ParseStream) -> Result<Self> {
        let content;
        Ok(Self {
            path: input.parse()?,
            brace_token: braced!(content in input),
            args: content.parse()?,
        })
    }
}

impl ViewElement {
    pub fn quote(&self) -> TokenStream {
        let Self { path, args, .. } = self;
        let builtin = path.get_ident().filter(|&ident| {
            ident
                .to_string()
                .chars()
                .all(|c| matches!(c, 'a'..='z' | '0'..='9' | '_'))
        });
        let props = args.quote_props();
        let children = args.quote_children();
        if let Some(builtin) = builtin {
            quote!({ #FN_VIEW_ELEMENT(
                #VAR_CX,
                #M_ELEMENTS::#builtin,
                move |#VAR_VIEW| { #VAR_VIEW #props },
                move |#VAR_VIEW| { #VAR_VIEW #children },
            ) })
        } else {
            quote!({ #FN_VIEW_COMPONENT(
                #VAR_CX,
                #path,
                move |#VAR_VIEW| { #VAR_VIEW #props },
                move |#VAR_VIEW| { #VAR_VIEW #children },
            ) })
        }
    }
}

fn quote_children(children: &[ViewChild]) -> TokenStream {
    let children = children.iter().map(ViewChild::quote);
    let fn_ = FN_CHILD;
    quote!(#(.#fn_(#children))*)
}
