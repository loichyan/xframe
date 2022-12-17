use super::Visit;

pub struct VisitSkip<'a, V: ?Sized> {
    pub(crate) visitor: &'a V,
    pub(crate) count: usize,
}

impl<'a, T, V> Visit<T> for VisitSkip<'a, V>
where
    V: Visit<T>,
{
    fn visit(&self, mut f: impl FnMut(&T)) {
        let mut visited = 0;
        self.visitor.visit(|t| {
            if visited >= self.count {
                f(t);
            }
            visited += 1;
        })
    }

    fn count(&self) -> usize {
        V::count(self.visitor) - self.count
    }
}
