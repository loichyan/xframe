/*
  The code is heavily inspired by the [`slotmap`] codebase.

  [`slotmap`]: https://github.com/orlp/slomap

  Copyright (c) 2021 Orson Peters <orsonpeters@gmail.com>

  This software is provided 'as-is', without any express or implied warranty. In
  no event will the authors be held liable for any damages arising from the use of
  this software.

  Permission is granted to anyone to use this software for any purpose, including
  commercial applications, and to alter it and redistribute it freely, subject to
  the following restrictions:

   1. The origin of this software must not be misrepresented; you must not claim
      that you wrote the original software. If you use this software in a product,
      an acknowledgment in the product documentation would be appreciated but is
      not required.

   2. Altered source versions must be plainly marked as such, and must not be
      misrepresented as being the original software.

   3. This notice may not be removed or altered from any source distribution.
*/

use bumpalo::Bump;
use std::{
    cell::{Cell, RefCell},
    hash::Hash,
    mem::ManuallyDrop,
    ptr::NonNull,
};

type FreeHead<T> = Option<NonNull<Slot<T>>>;

/// A slot-based bump allocation arena.
///
/// This is mainly for:
///
/// 1. Allocated objects during the same program phase, used, and then deallocate
/// them together as a group;
/// 2. Reuse a free slot to avoid unnecessary memory allocation.
pub(crate) struct Arena<T> {
    bump: Bump,
    free_head: Cell<FreeHead<T>>,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Arena {
            bump: Default::default(),
            free_head: Default::default(),
        }
    }
}

impl<T> Arena<T> {
    pub fn alloc(&self, t: T) -> WeakRef<'_, T> {
        self.alloc_with_weak(|_| t)
    }

    /// Allocate a value with the reference to be returned (the reference can't
    /// be upgraded due to mismatched versions). This will first look up for an
    /// allocated freed slot before allocating a new one.
    pub fn alloc_with_weak<'a>(&'a self, f: impl FnOnce(WeakRef<'a, T>) -> T) -> WeakRef<'a, T> {
        let free_head = self.free_head.get().map(|addr| unsafe { addr.as_ref() });

        // 1) Get a freed slot.
        let slot = free_head.unwrap_or_else(|| {
            // Create a vacant Slot
            let version = Cell::new(0);
            let content = RefCell::new(Content { next_free: None });
            self.bump.alloc(Slot { version, content })
        });

        // 2) Get the value.
        let version = slot.version.get() | 1;
        let weak = WeakRef { version, slot };
        let val = f(weak);

        // 3) Bump version.
        debug_assert_eq!(slot.version.get() + 1, version);
        slot.version.set(version);

        // 4) Update free head.
        let content = &mut *slot.content.borrow_mut();
        let next_free = unsafe { content.next_free };
        self.free_head.set(next_free);

        // 5) Write value.
        content.value = ManuallyDrop::new(val);

        weak
    }

    /// # Panic
    ///
    /// Panics on a borrowed slot.
    pub fn free(&self, weak: WeakRef<'_, T>) -> bool {
        self.try_free(weak)
            .unwrap_or_else(|_| panic!("free a in using slot"))
    }

    /// Try to free a slot, raise an error if the value is borrowed. It will
    /// do nothing if the reference is outdate.
    pub fn try_free(&self, weak: WeakRef<'_, T>) -> Result<bool, ()> {
        // Check the ownship.
        if !weak.can_upgrade() {
            return Ok(false);
        }
        let slot = weak.slot;
        let content = &mut *slot.content.try_borrow_mut().map_err(|_| ())?;

        // 1) Bump version.
        debug_assert!(slot.version.get() % 2 > 0);
        slot.version.set(slot.version.get() + 1);

        // 2) Call the destructor.
        unsafe { ManuallyDrop::drop(&mut content.value) };

        // 3) Update free head.
        content.next_free = self.free_head.get();
        self.free_head.set(Some(NonNull::from(slot)));

        Ok(true)
    }
}

struct Slot<T> {
    version: Cell<u32>, // even => vacant, odd => occupied
    content: RefCell<Content<T>>,
}

union Content<T> {
    next_free: FreeHead<T>,
    value: ManuallyDrop<T>,
}

/// A weak reference points to a value allocated by the [`Arena`], once the slot
/// get freed, it can't be upgraded to a normal reference anymore.
pub(crate) struct WeakRef<'a, T> {
    version: u32,
    slot: &'a Slot<T>,
}

impl<T> Clone for WeakRef<'_, T> {
    fn clone(&self) -> Self {
        WeakRef {
            version: self.version,
            slot: self.slot,
        }
    }
}

impl<T> Copy for WeakRef<'_, T> {}

impl<T> PartialEq for WeakRef<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_tuple().eq(&other.as_tuple())
    }
}

impl<T> Eq for WeakRef<'_, T> {}

impl<T> Hash for WeakRef<'_, T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_tuple().hash(state)
    }
}

impl<'a, T> WeakRef<'a, T> {
    fn as_tuple(&self) -> (u32, *const Slot<T>) {
        (self.version, self.slot as *const Slot<T>)
    }

    pub fn as_ptr(&self) -> *const T {
        let ptr: &T = unsafe { &self.slot.content.borrow().value };
        ptr as *const T
    }

    pub fn can_upgrade(&self) -> bool {
        self.slot.version.get() == self.version
    }

    pub fn with<U>(&self, f: impl FnOnce(&T) -> U) -> Option<U> {
        if !self.can_upgrade() {
            return None;
        }
        let val = unsafe { &self.slot.content.borrow().value };
        Some(f(val))
    }

    /// Leak the underlying content of the slot which this reference points to.
    ///
    /// # Safety
    ///
    /// The slot can be freed and allocated for a new value, therefore it's unsafe
    /// to read the reference.
    pub unsafe fn leak_ref(self) -> &'a T {
        &(*self.slot.content.as_ptr()).value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_and_free() {
        thread_local! {
            static COUNTER: Cell<usize> = Cell::new(0);
        }

        struct DropMe;

        impl Drop for DropMe {
            fn drop(&mut self) {
                COUNTER.with(|x| x.set(x.get() + 1));
            }
        }

        let arena = Arena::<DropMe>::default();
        let val1 = arena.alloc(DropMe);
        let val2 = arena.alloc(DropMe);
        COUNTER.with(|x| assert_eq!(x.get(), 0));
        arena.free(val1);
        COUNTER.with(|x| assert_eq!(x.get(), 1));
        arena.free(val2);
        COUNTER.with(|x| assert_eq!(x.get(), 2));
    }

    #[test]
    fn upgrade_weak_ref() {
        let arena = Arena::<i32>::default();
        let val = arena.alloc(0);

        assert!(val.can_upgrade());
        arena.free(val);
        assert!(!val.can_upgrade());
    }

    #[test]
    fn reuse_a_slot() {
        let arena = Arena::<i32>::default();

        let val1 = arena.alloc(0);
        let addr1 = val1.slot as *const _;

        val1.with(|x| assert_eq!(*x, 0));
        arena.free(val1);

        let val2 = arena.alloc(1);
        let addr2 = val2.slot as *const _;

        val2.with(|x| assert_eq!(*x, 1));
        assert_eq!(addr1, addr2);
    }

    #[test]
    fn cannot_free_a_in_borrowed_weak() {
        let arena = Arena::<i32>::default();
        let val = arena.alloc(0);

        val.with(|_| {
            assert!(arena.try_free(val).is_err());
        });
    }

    #[test]
    fn cannot_upgrade_a_weak_before_alloc() {
        let arena = Arena::<i32>::default();
        arena.alloc_with_weak(|weak| {
            assert!(!weak.can_upgrade());
            0
        });
    }

    #[test]
    fn cannot_free_a_reused_slot() {
        let arena = Arena::<i32>::default();
        let val1 = arena.alloc(0);
        assert!(arena.free(val1));
        let val2 = arena.alloc(1);
        assert!(!arena.free(val1));
        assert!(arena.free(val2));
    }
}
