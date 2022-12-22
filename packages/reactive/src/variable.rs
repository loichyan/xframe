use crate::{
    runtime::{VariableId, RT},
    scope::{Cleanup, Scope},
    ThreadLocal,
};
use std::{any::Any, fmt, marker::PhantomData};

pub struct Variable<T: 'static> {
    pub(crate) id: VariableId,
    pub(crate) marker: PhantomData<(T, ThreadLocal)>,
}

impl<T> Clone for Variable<T> {
    fn clone(&self) -> Self {
        Self { ..*self }
    }
}

impl<T> Copy for Variable<T> {}

impl<T: fmt::Debug> fmt::Debug for Variable<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.read(|v| f.debug_tuple("Variable").field(v).finish())
    }
}

impl<T: fmt::Display> fmt::Display for Variable<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.read(|v| v.fmt(f))
    }
}

impl<T> Variable<T> {
    pub fn read<U>(&self, f: impl FnOnce(&T) -> U) -> U {
        RT.with(|rt| {
            f(rt.variables
                .borrow()
                .get(self.id)
                .unwrap_or_else(|| panic!("tried to access a disposed variable"))
                .downcast_ref()
                .unwrap_or_else(|| panic!("tried to use a variable in mismatched types")))
        })
    }

    pub fn get(&self) -> T
    where
        T: Clone,
    {
        self.read(T::clone)
    }

    pub fn write<U>(&self, f: impl FnOnce(&mut T) -> U) -> U {
        RT.with(|rt| {
            f(rt.variables
                .borrow_mut()
                .get_mut(self.id)
                .unwrap_or_else(|| panic!("tried to access a disposed variable"))
                .downcast_mut()
                .unwrap_or_else(|| panic!("tried to use a variable in mismatched types")))
        })
    }

    pub fn set(&self, t: T) {
        self.write(|v| *v = t);
    }

    pub fn update(&self, f: impl FnOnce(&T) -> T) {
        self.write(|t| *t = f(t));
    }
}

impl Scope {
    fn create_variable_dyn(&self, t: Box<dyn Any>) -> VariableId {
        self.with_shared(|rt| {
            self.id.with(rt, |cx| {
                let id = rt.variables.borrow_mut().insert(t);
                cx.push_cleanup(Cleanup::Variable(id));
                id
            })
        })
    }

    pub fn create_variable<T>(&self, t: T) -> Variable<T> {
        Variable {
            id: self.create_variable_dyn(Box::new(t)),
            marker: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_root;

    #[test]
    fn variable() {
        create_root(|cx| {
            let var = cx.create_variable(1);
            assert_eq!(var.get(), 1);
            var.set(2);
            assert_eq!(var.get(), 2);
        });
    }

    #[test]
    #[should_panic = "tried to access a disposed variable"]
    fn cannot_read_disposed_variables() {
        struct DropAndRead(Option<Variable<i32>>);
        impl Drop for DropAndRead {
            fn drop(&mut self) {
                self.0.unwrap().get();
            }
        }

        create_root(|cx| {
            let var1 = cx.create_variable(DropAndRead(None));
            let var2 = cx.create_variable(0);
            var1.write(|v| v.0 = Some(var2));
        });
    }

    #[test]
    fn access_previous_variable_on_drop() {
        struct DropAndAssert {
            var: Variable<i32>,
            expect: i32,
        }
        impl Drop for DropAndAssert {
            fn drop(&mut self) {
                assert_eq!(self.var.get(), self.expect);
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
