#[macro_use]
mod utils;
mod view;

use proc_macro::TokenStream;
use syn::parse_macro_input;

#[proc_macro]
pub fn view(input: TokenStream) -> TokenStream {
    parse_macro_input!(input with view::expand).into()
}
