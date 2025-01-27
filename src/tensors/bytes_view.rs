use std::ops::{Deref, DerefMut};

use bytes::Bytes;

use crate::tensors::GpuNum;

/// for splitting big memory mapped region into smaller arrays of concrete type
#[derive(Clone, Eq, Hash, PartialEq)]
pub struct BytesView<T: GpuNum> {
    repr: ViewRepr<T>,
}
#[derive(Clone, Eq, Hash, PartialEq)]
enum ViewRepr<T: GpuNum> {
    Box(Box<[T]>),
    Mapped(Bytes, usize),
}

impl<T: GpuNum> BytesView<T> {
    pub fn from_slice(data: Box<[T]>) -> Self {
        // struct BoxOwner<T: GpuNum>(Box<[T]>);
        // impl<T: GpuNum> AsRef<[u8]> for BoxOwner<T> {
        //     fn as_ref(&self) -> &[u8] {
        //         // SAFETY: T is GpuNum which is sealed and all field patterns are valid in it
        //         unsafe { self.0.align_to().1 }
        //     }
        // }
        // let data = Bytes::from_owner(BoxOwner(data));
        // Self::new(data) // data is already aligned
        Self { repr: ViewRepr::Box(data) }
    }
    pub fn new_mapped(data: Bytes) -> Self {
        match Self::try_new_mapped(data) {
            Ok(v) => v,
            Err(unalign) => panic!("data is not properly aligned, unaligned by {unalign} bytes"),
        }
    }
    pub fn try_new_mapped(data: Bytes) -> Result<Self, usize> {
        // SAFETY: T is GpuNum which is sealed and all field patterns are valid in it
        let (prev, aligned_items, _) = unsafe { data.align_to::<T>() };
        if !prev.is_empty() {
            return Err(prev.len());
        }
        let aligned_items = aligned_items.len();
        if aligned_items == 0 {
            return Ok(Self {
                repr: ViewRepr::Mapped(Bytes::new(), 0),
            });
        }
        Ok(Self {
            repr: ViewRepr::Mapped(data, aligned_items),
        })
    }
    #[inline]
    pub const fn len(&self) -> usize {
        match &self.repr {
            ViewRepr::Mapped(_, items) => *items,
            ViewRepr::Box(data) => data.len(),
        }
    }
    #[inline]
    pub const fn is_empty(&self) -> bool {
        match &self.repr {
            ViewRepr::Mapped(_, items) => *items == 0,
            ViewRepr::Box(data) => data.is_empty(),
        }
    }

    pub fn upgrade_mut(&mut self) {
        if matches!(self.repr, ViewRepr::Mapped(_, _)) {
            self.repr = ViewRepr::Box(self.deref().to_vec().into_boxed_slice());
        }
    }
}

impl<T: GpuNum> Deref for BytesView<T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        match &self.repr {
            // SAFETY: T is GpuNum which is sealed and all bit patterns are valid in it, we checked items size and align of data in try_new
            ViewRepr::Mapped(data, items) => unsafe {
                let ptr = data.as_ptr() as *const T;
                debug_assert!(ptr.is_aligned());
                std::slice::from_raw_parts(ptr, *items)
            },
            ViewRepr::Box(data) => data,
        }
    }
}

impl<T: GpuNum> DerefMut for BytesView<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.upgrade_mut();
        match &mut self.repr {
            ViewRepr::Box(data) => data,
            _ => unreachable!(),
        }
    }
}

impl<T: GpuNum> std::fmt::Debug for BytesView<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <[T] as std::fmt::Debug>::fmt(self, f)
    }
}
