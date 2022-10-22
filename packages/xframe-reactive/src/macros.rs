macro_rules! impl_clone_copy {
    ($name:ident[$($param:tt)*]) => {
        impl <$($param)*> Clone for $name<$($param)*> {
            fn clone(&self) -> Self {
                $name { inner: self.inner }
            }
        }

        impl <$($param)*> Copy for $name<$($param)*> { }
    };
}
