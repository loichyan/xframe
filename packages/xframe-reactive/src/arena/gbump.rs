use bumpalo::Bump;
use std::{
    cell::{Cell, UnsafeCell},
    hash::Hash,
    mem::ManuallyDrop,
    ops::Deref,
    ptr::NonNull,
};

type FreeHead<T> = Option<NonNull<Slot<T>>>;

/// Generational bump allocation arena.
///
/// Based on <https://github.com/orlp/slotmap>.
pub struct GBump<T> {
    bump: Bump,
    free_head: Cell<FreeHead<T>>,
}

impl<T> Default for GBump<T> {
    fn default() -> Self {
        GBump {
            bump: Default::default(),
            free_head: Default::default(),
        }
    }
}

impl<T> GBump<T> {
    pub fn alloc(&self, t: T) -> Owned<'_, T> {
        self.alloc_with_weak(|_| t)
    }

    pub fn alloc_with_weak<'a>(&'a self, f: impl FnOnce(WeakRef<'a, T>) -> T) -> Owned<'a, T> {
        let free_head = self.free_head.get().map(|addr| unsafe { addr.as_ref() });

        // 0) get slot
        let slot = free_head.unwrap_or_else(|| {
            // create a vacant Slot
            let version = Cell::new(0);
            let content = UnsafeCell::new(Content { next_free: None });
            self.bump.alloc(Slot { version, content })
        });

        // 1) get value
        let version = slot.version.get() | 1;
        let weak = WeakRef { version, slot };
        let t = f(weak);

        // 2) bump version
        debug_assert_eq!(slot.version.get() + 1, version);
        slot.version.set(version);

        // 3) update free head
        let content = unsafe { &mut *slot.content.get() };
        let next_free = unsafe { content.next_free };
        self.free_head.set(next_free);

        // 4) write value
        content.value = ManuallyDrop::new(t);

        // 5) return
        Owned { slot, bump: self }
    }

    unsafe fn free(&self, slot: &Slot<T>) {
        // 1) bump version
        debug_assert!(slot.version.get() % 2 > 0);
        slot.version.set(slot.version.get() + 1);

        // 2) call the destructor
        let content = &mut *slot.content.get();
        ManuallyDrop::drop(&mut content.value);

        // 3) update free head
        content.next_free = self.free_head.get();
        self.free_head.set(Some(NonNull::from(slot)));
    }
}

struct Slot<T> {
    version: Cell<u32>, // even => vacant, odd => occupied
    content: UnsafeCell<Content<T>>,
}

impl<T> Slot<T> {
    fn get_value(&self) -> &T {
        debug_assert!(self.version.get() % 2 > 0);
        unsafe { &(&*self.content.get()).value }
    }

    fn upgrade(&self, version: u32) -> Option<&T> {
        if version == self.version.get() {
            Some(self.get_value())
        } else {
            None
        }
    }
}

union Content<T> {
    next_free: FreeHead<T>,
    value: ManuallyDrop<T>,
}

pub struct Owned<'a, T> {
    bump: &'a GBump<T>,
    slot: &'a Slot<T>,
}

impl<T> Drop for Owned<'_, T> {
    fn drop(&mut self) {
        unsafe { self.bump.free(self.slot) };
    }
}

impl<T> Deref for Owned<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.slot.get_value()
    }
}

impl<'a, T> Owned<'a, T> {
    pub fn downgrade(&self) -> WeakRef<'a, T> {
        WeakRef {
            version: self.slot.version.get(),
            slot: self.slot,
        }
    }
}

pub struct WeakRef<'a, T> {
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

    #[cfg(test)]
    pub fn upgrade(&self) -> Option<Ref<'_, T>> {
        self.slot.upgrade(self.version).map(Ref)
    }

    pub fn with<U>(&self, f: impl FnOnce(&T) -> U) -> Option<U> {
        self.slot.upgrade(self.version).map(f)
    }
}

pub struct Ref<'a, T>(&'a T);

impl<T> Deref for Ref<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.0
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

        let bump = GBump::<DropMe>::default();
        let val1 = bump.alloc(DropMe);
        let val2 = bump.alloc(DropMe);
        COUNTER.with(|x| assert_eq!(x.get(), 0));
        drop(val1);
        COUNTER.with(|x| assert_eq!(x.get(), 1));
        drop(val2);
        COUNTER.with(|x| assert_eq!(x.get(), 2));
    }

    #[test]
    fn upgrade_weak_ref() {
        let bump = GBump::<i32>::default();
        let val1 = bump.alloc(0);
        let ref1 = Owned::downgrade(&val1);

        assert!(ref1.upgrade().is_some());
        drop(val1);
        assert!(ref1.upgrade().is_none());
    }

    #[test]
    fn reuse_a_slot() {
        let bump = GBump::<i32>::default();

        let val1 = bump.alloc(0);
        let addr1 = val1.slot as *const _;

        assert_eq!(*val1, 0);
        drop(val1);

        let val2 = bump.alloc(1);
        let addr2 = val2.slot as *const _;

        assert_eq!(*val2, 1);
        assert_eq!(addr1, addr2);
    }

    #[test]
    fn ungrade_weak_in_alloc_is_none() {
        let bump = GBump::<i32>::default();
        bump.alloc_with_weak(|weak| {
            assert!(weak.upgrade().is_none());
            0
        });
    }
}
