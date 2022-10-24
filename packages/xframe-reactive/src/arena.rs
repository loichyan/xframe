use bumpalo::Bump;
use slotmap::{DefaultKey, SlotMap};
use smallvec::SmallVec;
use std::{cell::RefCell, fmt, marker::PhantomData};

use crate::scope::ScopeShared;

const INITIALIAL_VARIABLE_SLOTS: usize = 4;

trait Empty {}
impl<T> Empty for T {}

#[derive(Default)]
pub(crate) struct Arena<'a> {
    bump: Bump,
    variables: RefCell<SmallVec<[&'a dyn Empty; INITIALIAL_VARIABLE_SLOTS]>>,
}

impl fmt::Debug for Arena<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(
                self.variables
                    .borrow()
                    .iter()
                    .map(|x| x as *const dyn Empty),
            )
            .finish()
    }
}

impl<'a> Arena<'a> {
    pub fn alloc<T: 'a>(&'a self, t: T) -> &'a T {
        let val = &*self.bump.alloc(t);
        // SAFETY: The returned reference is bound by 'a, and the pointer is
        // only used to call the destructor.
        let ptr = val as &dyn Empty;
        self.variables.borrow_mut().push(ptr);
        val
    }

    pub fn alloc_in_slots<T: 'a>(&'a self, shared: &'a ScopeShared, t: T) -> &'a SlotValue<'a, T> {
        let mut slot = None;
        shared.slots.inner.borrow_mut().insert_with_key(|id| {
            let key = SlotKey {
                id,
                ty: PhantomData,
            };
            let val = SlotValue {
                key,
                shared,
                value: t,
            };
            let ptr = self.alloc(val);
            slot = Some(ptr);
            // SAFETY: The type is constraint by SlotKey, and it can't be accessed
            // once the SlotValue is disposed.
            unsafe { std::mem::transmute(ptr as *const dyn Empty) }
        });
        slot.unwrap_or_else(|| unreachable!())
    }

    pub unsafe fn dispose(&self) {
        // SAFETY: last alloced variables must be disposed first because signals
        // and effects need to do some cleanup works with its captured references.
        for ptr in self.variables.take().into_iter().rev() {
            std::ptr::drop_in_place(ptr as *const dyn Empty as *mut dyn Empty);
        }
    }
}

pub(crate) struct SlotKey<T> {
    id: DefaultKey,
    ty: PhantomData<T>,
}

#[derive(Default)]
pub(crate) struct Slots {
    inner: RefCell<SlotMap<DefaultKey, *const dyn Empty>>,
}

impl Slots {
    pub fn get<T>(&self, key: SlotKey<T>) -> Option<&T> {
        self.inner
            .borrow()
            .get(key.id)
            .copied()
            .map(|ptr| unsafe { &*(ptr as *const T) })
    }
}

pub(crate) struct SlotValue<'a, T> {
    pub key: SlotKey<T>,
    pub shared: &'a ScopeShared,
    pub value: T,
}

impl<T> Drop for SlotValue<'_, T> {
    fn drop(&mut self) {
        self.shared.slots.inner.borrow_mut().remove(self.key.id);
    }
}
