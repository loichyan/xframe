use bumpalo::Bump;

#[derive(Debug, Default)]
pub(crate) struct Arena {
    bump: Bump,
}

impl Arena {
    pub fn alloc<'a, T: 'a>(&'a self, t: T) -> (&'a T, Disposer) {
        let val = &*self.bump.alloc(t);
        // SAFETY: The returned reference is bound by 'a, and the pointer is
        // only used to call the destructor.
        let ptr = val as *const dyn Empty as *mut dyn Empty;
        (val, Disposer(ptr))
    }
}

trait Empty {}
impl<T> Empty for T {}

#[derive(Debug)]
pub(crate) struct Disposer(*mut dyn Empty);

impl Disposer {
    pub unsafe fn dispose(&mut self) {
        let val = &mut *self.0;
        std::ptr::drop_in_place(val);
    }
}
