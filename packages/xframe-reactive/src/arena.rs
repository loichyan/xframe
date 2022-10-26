use bumpalo::Bump;
use std::{
    cell::{Cell, RefCell},
    hash::Hash,
    mem::ManuallyDrop,
    ptr::NonNull,
};

type FreeHead<T> = Option<NonNull<Slot<T>>>;

/// Generational bump allocation arena.
///
/// Based on <https://github.com/orlp/slotmap>.
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
        let free_head = self.free_head.get().map(|addr| unsafe { addr.as_ref() });

        // 1) get slot
        let slot = free_head.unwrap_or_else(|| {
            // create a vacant Slot
            let version = Cell::new(0);
            let content = RefCell::new(Content { next_free: None });
            self.bump.alloc(Slot { version, content })
        });

        // 2) bump version
        let version = slot.version.get() | 1;
        debug_assert_eq!(slot.version.get() + 1, version);
        slot.version.set(version);

        // 3) update free head
        let content = &mut *slot.content.borrow_mut();
        let next_free = unsafe { content.next_free };
        self.free_head.set(next_free);

        // 4) write value
        content.value = ManuallyDrop::new(t);

        // 5) return
        WeakRef { version, slot }
    }

    pub fn free(&self, weak: WeakRef<'_, T>) -> bool {
        self.try_free(weak)
            .unwrap_or_else(|_| panic!("free a in using slot"))
    }

    pub fn try_free(&self, weak: WeakRef<'_, T>) -> Result<bool, ()> {
        // Check the ownship.
        if !weak.can_upgrade() {
            return Ok(false);
        }
        let slot = weak.slot;
        let content = &mut *slot.content.try_borrow_mut().map_err(|_| ())?;

        // 1) bump version
        debug_assert!(slot.version.get() % 2 > 0);
        slot.version.set(slot.version.get() + 1);

        // 2) call the destructor
        unsafe { ManuallyDrop::drop(&mut content.value) };

        // 3) update free head
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
    fn cannot_free_a_reused_slot() {
        let arena = Arena::<i32>::default();
        let val1 = arena.alloc(0);
        assert!(arena.free(val1));
        let val2 = arena.alloc(1);
        assert!(!arena.free(val1));
        assert!(arena.free(val2));
    }
}
