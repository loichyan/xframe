use proc_macro2::TokenStream;
use quote::ToTokens;
use syn::token;

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

pub struct QuoteSurround<S, T>(pub S, pub T);

pub trait Surround {
    fn surround(&self, tokens: &mut TokenStream, f: impl FnOnce(&mut TokenStream));
}

impl<T: Surround> Surround for &T {
    fn surround(&self, tokens: &mut TokenStream, f: impl FnOnce(&mut TokenStream)) {
        T::surround(self, tokens, f);
    }
}

macro_rules! impl_surround {
    ($($ty:ident),*) => {$(
        impl Surround for token::$ty {
            fn surround(&self, tokens: &mut TokenStream, f: impl FnOnce(&mut TokenStream)) {
                token::$ty::surround(self, tokens, f);
            }
        }
    )*};
}

impl_surround!(Paren, Brace, Bracket);

impl<S: Surround, T: ToTokens> ToTokens for QuoteSurround<S, T> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.0.surround(tokens, |tokens| self.1.to_tokens(tokens));
    }
}
