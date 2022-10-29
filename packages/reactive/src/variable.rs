use crate::OwnedScope;
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

pub type Variable<'a, T> = &'a OwnedVariable<'a, T>;

pub struct OwnedVariable<'a, T> {
    disposed: Cell<bool>,
    value: RefCell<T>,
    bounds: PhantomData<&'a ()>,
}

impl<T> Drop for OwnedVariable<'_, T> {
    fn drop(&mut self) {
        if self.value.try_borrow_mut().is_err() {
            panic!("dispose a borrowed variable");
        }
        self.disposed.set(true);
    }
}

impl<'a, T> OwnedVariable<'a, T> {
    pub fn is_disposed(&self) -> bool {
        self.disposed.get()
    }

    pub fn get(&self) -> VarRef<'_, T> {
        self.try_get().expect("get a disposed varaible")
    }

    pub fn try_get(&self) -> Option<VarRef<'_, T>> {
        if self.is_disposed() {
            return None;
        }
        Some(VarRef(self.value.borrow()))
    }

    pub fn get_mut(&self) -> VarRefMut<'_, T> {
        self.try_get_mut().expect("get a disposed variable")
    }

    pub fn try_get_mut(&self) -> Option<VarRefMut<'_, T>> {
        if self.is_disposed() {
            return None;
        }
        Some(VarRefMut(self.value.borrow_mut()))
    }
}

pub struct VarRef<'a, T>(Ref<'a, T>);

impl<'a, T> From<&'a RefCell<T>> for VarRef<'a, T> {
    fn from(r: &'a RefCell<T>) -> Self {
        VarRef(r.borrow())
    }
}

impl<'a, T> VarRef<'a, T> {
    pub fn map<U>(orig: Self, f: impl FnOnce(&T) -> &U) -> VarRef<'a, U> {
        VarRef(Ref::map(orig.0, f))
    }
}

impl<T> Deref for VarRef<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct VarRefMut<'a, T>(RefMut<'a, T>);

impl<'a, T> From<&'a RefCell<T>> for VarRefMut<'a, T> {
    fn from(r: &'a RefCell<T>) -> Self {
        VarRefMut(r.borrow_mut())
    }
}

impl<T> Deref for VarRefMut<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for VarRefMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, T> VarRefMut<'a, T> {
    pub fn map<U>(orig: Self, f: impl FnOnce(&mut T) -> &mut U) -> VarRefMut<'a, U> {
        VarRefMut(RefMut::map(orig.0, f))
    }
}

impl<'a> OwnedScope<'a> {
    pub fn create_owned_variable<T>(&'a self, t: T) -> OwnedVariable<'a, T> {
        OwnedVariable {
            disposed: Cell::new(false),
            value: RefCell::new(t),
            bounds: PhantomData,
        }
    }

    pub fn create_variable<T>(&'a self, t: T) -> Variable<'a, T> {
        let owned = self.create_owned_variable(t);
        unsafe { self.create_variable_unchecked(owned) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_root;
    use std::cell::Cell;

    #[test]
    fn cannot_read_disposed_variable() {
        struct DropAndRead<'a> {
            ref_to: Cell<Option<Variable<'a, String>>>,
        }
        impl Drop for DropAndRead<'_> {
            fn drop(&mut self) {
                assert!(self.ref_to.get().unwrap().try_get().is_none());
            }
        }

        create_root(|cx| {
            let var1 = cx.create_variable(DropAndRead {
                ref_to: Default::default(),
            });
            let var2 = cx.create_variable(String::from("Hello, xFrame!"));
            var1.get().ref_to.set(Some(var2));
        });
    }

    #[test]
    #[should_panic = "dispose a borrowed variable"]
    fn cannot_dispose_borrowed_varialbe() {
        create_root(|cx| {
            let var = cx.create_variable(None);
            let var2 = cx.create_variable(0);
            *var.get_mut() = Some(var2.get());
        });
    }
}
