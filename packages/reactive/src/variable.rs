use crate::{
    scope::Scope,
    shared::{VariableId, SHARED},
    CovariantLifetime,
};
use std::{
    cell::{Ref, RefCell, RefMut},
    fmt,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

pub struct Variable<'a, T> {
    pub(crate) id: VariableId,
    marker: PhantomData<(T, CovariantLifetime<'a>)>,
}

impl<T> Clone for Variable<'_, T> {
    fn clone(&self) -> Self {
        Self { ..*self }
    }
}

impl<T> Copy for Variable<'_, T> {}

impl<T: fmt::Debug> fmt::Debug for Variable<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Variable").field(&*self.get()).finish()
    }
}

impl<T: fmt::Display> fmt::Display for Variable<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.get().fmt(f)
    }
}

impl<'a, T> Variable<'a, T> {
    fn value(&self) -> &'a VarSlot<T> {
        SHARED.with(|shared| {
            let ptr = shared
                .variables
                .borrow()
                .get(self.id)
                .copied()
                .unwrap_or_else(|| panic!("tried to access a disposed variable"));
            // SAFETY: The type is assumed by the marker `ty` and the allocated variable
            // lives as long as current `Scope`.
            unsafe { NonNull::from(ptr).cast().as_ref() }
        })
    }

    pub fn get(&self) -> VarRef<'_, T> {
        self.value().get()
    }

    pub fn get_mut(&self) -> VarRefMut<'_, T> {
        self.value().get_mut()
    }
}

/// A wrapper ensures borrowed variables cannot be disposed. Most values allocated
/// in the arena of `Scope`s should use this to avoid undefined behaviors.
pub(crate) struct VarSlot<T>(RefCell<T>);

impl<T> Drop for VarSlot<T> {
    fn drop(&mut self) {
        if self.0.try_borrow_mut().is_err() {
            panic!("tried to dispose a borrowed value")
        }
    }
}

impl<T> VarSlot<T> {
    pub fn new(t: T) -> Self {
        Self(RefCell::new(t))
    }

    pub fn get(&self) -> VarRef<'_, T> {
        VarRef(self.0.borrow())
    }

    pub fn get_mut(&self) -> VarRefMut<'_, T> {
        VarRefMut(self.0.borrow_mut())
    }
}

pub struct VarRef<'a, T>(Ref<'a, T>);

impl<'a, T> VarRef<'a, T> {
    pub fn map<U>(this: Self, f: impl FnOnce(&T) -> &U) -> VarRef<'a, U> {
        VarRef(Ref::map(this.0, f))
    }
}

impl<T> Deref for VarRef<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct VarRefMut<'a, T>(RefMut<'a, T>);

impl<'a, T> VarRefMut<'a, T> {
    pub fn map<U>(this: Self, f: impl FnOnce(&mut T) -> &mut U) -> VarRefMut<'a, U> {
        VarRefMut(RefMut::map(this.0, f))
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

impl VariableId {
    pub(crate) unsafe fn bound<'a, T>(&self) -> Variable<'a, T> {
        Variable {
            id: *self,
            marker: PhantomData,
        }
    }
}

impl<'a> Scope<'a> {
    pub fn create_variable<T: 'a>(&self, t: T) -> Variable<'a, T> {
        self.with_shared(|shared| {
            self.id.with(shared, |cx| {
                let id = cx.alloc_variable(shared, t);
                Variable {
                    id,
                    marker: PhantomData,
                }
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_root;

    #[test]
    fn map_var_ref() {
        create_root(|cx| {
            let var = cx.create_variable((1, 2, 3));
            let var_ref = var.get();
            assert_eq!(&*var_ref, &(1, 2, 3));
            let field1 = VarRef::map(var_ref, |(f1, _, _)| f1);
            assert_eq!(*field1, 1);
        });
    }

    #[test]
    #[should_panic = "tried to access a disposed variable"]
    fn cannot_read_disposed_variables() {
        struct DropAndRead<'a>(Option<Variable<'a, i32>>);
        impl Drop for DropAndRead<'_> {
            fn drop(&mut self) {
                self.0.unwrap().get();
            }
        }

        create_root(|cx| {
            let var1 = cx.create_variable(DropAndRead(None));
            let var2 = cx.create_variable(0);
            (*var1.get_mut()).0 = Some(var2);
        });
    }

    #[test]
    #[should_panic = "tried to dispose a borrowed value"]
    fn cannot_dispose_borrowed_varialbes() {
        create_root(|cx| {
            let trigger = cx.create_signal(());
            let store_disposer = cx.create_variable(None);
            let disposer = cx.create_child(|cx| {
                cx.create_effect(move |_| {
                    trigger.track();
                    if let Some(disposer) = store_disposer.get_mut().take() {
                        // Panics because the effect (`VarSlot<AnyEffectImpl>`)
                        // is borrowed and running.
                        drop(disposer);
                    }
                });
            });
            *store_disposer.get_mut() = Some(disposer);
            trigger.trigger();
        });
    }

    #[test]
    fn access_previous_variable_on_drop() {
        struct DropAndAssert<'a> {
            var: Variable<'a, i32>,
            expect: i32,
        }
        impl Drop for DropAndAssert<'_> {
            fn drop(&mut self) {
                assert_eq!(*self.var.get(), self.expect);
            }
        }

        create_root(|cx| {
            let var = cx.create_variable(777);
            cx.create_variable(DropAndAssert { var, expect: 777 });
        });
    }

    #[test]
    fn fmt_variable() {
        create_root(|cx| {
            let var = cx.create_variable(0);
            assert_eq!(format!("{:?}", var), "Variable(0)");
            assert_eq!(format!("{}", var), "0");
        });
    }
}
