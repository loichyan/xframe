use bumpalo::Bump;
use smallvec::SmallVec;
use std::{cell::RefCell, fmt};

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

    pub unsafe fn dispose(&mut self) {
        // SAFETY: last alloced variables must be disposed first because signals
        // and effects need to do some cleanup works with its captured references.
        for ptr in self.variables.get_mut().iter().copied().rev() {
            std::ptr::drop_in_place(ptr as *const dyn Empty as *mut dyn Empty);
        }
    }
}
