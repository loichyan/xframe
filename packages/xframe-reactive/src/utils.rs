use std::{fmt, hash::Hash};

#[repr(transparent)]
pub(crate) struct ByAddress<'a, T: ?Sized>(pub &'a T);

impl<T: ?Sized> ByAddress<'_, T> {
    fn as_ptr(&self) -> *const T {
        self.0 as _
    }
}

impl<T: ?Sized> Clone for ByAddress<'_, T> {
    fn clone(&self) -> Self {
        ByAddress(self.0)
    }
}

impl<T: ?Sized> Copy for ByAddress<'_, T> {}

impl<T: ?Sized> fmt::Debug for ByAddress<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_ptr().fmt(f)
    }
}

impl<T: ?Sized> Hash for ByAddress<'_, T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_ptr().hash(state);
    }
}

impl<T: ?Sized> PartialEq for ByAddress<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_ptr().eq(&other.as_ptr())
    }
}

impl<T: ?Sized> Eq for ByAddress<'_, T> {}
