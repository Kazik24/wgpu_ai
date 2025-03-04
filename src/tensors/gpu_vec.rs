use std::{
    borrow::Cow,
    collections::HashMap,
    fmt::{Debug, Display},
    marker::PhantomData,
    mem::MaybeUninit,
    num::NonZeroU64,
    ops::{Bound, Range, RangeBounds},
    sync::{Arc, Condvar, Mutex},
};

use wgpu::{BufferUsages, CommandEncoderDescriptor, PipelineCompilationOptions, ShaderModuleDescriptor, SubmissionIndex, core::device, util::DeviceExt};

use super::{FlowFunc, GpuNum, WgpuContext};

pub struct GpuVec<T: GpuNum> {
    buffer: wgpu::Buffer,
    length: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: GpuNum> GpuVec<T> {
    pub fn with_usage(data: &[T], usage: BufferUsages) -> Self {
        let ctx = WgpuContext::get();
        let buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            // SAFETY: Any integer can be aligned to u8 without issue, we know data is of type GpuNum which is sealed and implemented only
            // by integers and floats which have all bit patters valid
            contents: unsafe { data.align_to::<u8>().1 },
            usage,
        });
        Self {
            buffer,
            length: data.len(),
            _phantom: PhantomData,
        }
    }

    pub fn new_storage(data: &[T]) -> Self {
        Self::with_usage(data, BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST)
    }
    pub fn new_uniform(data: &[T]) -> Self {
        Self::with_usage(data, BufferUsages::UNIFORM | BufferUsages::COPY_SRC | BufferUsages::COPY_DST)
    }

    pub fn empty_with_usage(length: usize, usage: BufferUsages) -> Self {
        let ctx = WgpuContext::get();
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: length as u64 * T::num_type().gpu_layout().size() as u64,
            usage,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            length,
            _phantom: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn gpu_bytes(&self) -> u64 {
        self.buffer.size()
    }

    pub fn write(&mut self, data: &[T]) {
        self.write_range(.., data);
    }
    pub fn write_range(&mut self, range: impl RangeBounds<usize>, data: &[T]) {
        let range = Self::check_bounds(range, self.len());
        assert!(data.len() == range.len(), "lengths must match");
        self.write_internal(range, data, true);
    }
    fn write_internal(&mut self, range: Range<usize>, data: &[T], submit: bool) {
        let range = Self::to_gpu_range(range);
        let Some(count) = NonZeroU64::new(range.clone().count() as u64) else {
            return;
        };

        let ctx = WgpuContext::get();

        let mut view = ctx.queue.write_buffer_with(&self.buffer, range.start, count).expect("Cannot write gpu buffer");
        // SAFETY: Any integer can be aligned to u8 without issue, we know data is of type GpuNum which is sealed and implemented only
        // by integers and floats which have all bit patters valid
        view.copy_from_slice(unsafe { data.align_to::<u8>().1 });
        if submit {
            ctx.queue.submit([]);
        }
    }

    fn check_bounds(range: impl RangeBounds<usize>, length: usize) -> Range<usize> {
        let start = match range.start_bound() {
            Bound::Included(start) => *start,
            Bound::Excluded(start) => *start + 1,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(end) => *end + 1,
            Bound::Excluded(end) => *end,
            Bound::Unbounded => length,
        };
        assert!(start <= end, "start must be lower or equal to end");
        assert!(end <= length, "end must be lower or equal to length");
        start..end
    }
    fn to_gpu_range(range: Range<usize>) -> Range<u64> {
        let start = range.start as u64 * T::num_type().gpu_layout().size() as u64;
        let end = range.end as u64 * T::num_type().gpu_layout().size() as u64;
        start..end
    }

    pub fn to_cpu(&self, range: impl RangeBounds<usize>) -> Vec<T> {
        let range = Self::check_bounds(range, self.len());
        self.to_cpu_internal(range)
    }

    pub fn rawr(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn usage(&self) -> BufferUsages {
        self.buffer.usage()
    }

    pub fn copy_to(&self, dst: &mut GpuVec<T>) {
        self.copy_ranges_to(.., .., dst);
    }
    pub fn copy_ranges_to(&self, src_range: impl RangeBounds<usize>, dst_range: impl RangeBounds<usize>, dst: &mut GpuVec<T>) {
        let src_range = Self::check_bounds(src_range, self.len());
        let dst_range = Self::check_bounds(dst_range, dst.len());
        assert!(dst_range.len() == src_range.len(), "lengths must match");
        if src_range.is_empty() {
            return;
        }
        self.copy_to_internal(src_range, dst_range, dst, true);
    }

    pub fn copy_to_internal(&self, src_range: Range<usize>, dst_range: Range<usize>, dst: &mut GpuVec<T>, poll: bool) -> Option<SubmissionIndex> {
        let src_range = Self::to_gpu_range(src_range);
        let copy_size = src_range.clone().count() as u64;
        let dst_offset = Self::to_gpu_range(dst_range).start;

        let ctx = WgpuContext::get();
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.buffer, src_range.start, &dst.buffer, dst_offset, copy_size);
        let index = ctx.queue.submit([encoder.finish()]);
        if poll {
            ctx.device.poll(wgpu::Maintain::WaitForSubmissionIndex(index));
            None
        } else {
            Some(index)
        }
    }

    // code based on: https://github.com/kurtschelfthout/tensorken/blob/5ad7cf15d39e947ac419192d55fbcabf86044b28/src/raw_tensor_wgpu.rs#L553
    fn to_cpu_internal(&self, range: Range<usize>) -> Vec<T> {
        if range.is_empty() {
            return Vec::new();
        }

        let len = range.len();
        let mut data = Vec::<T>::with_capacity(len);
        let to_fill = &mut data.spare_capacity_mut()[..len];
        self.copy_to_cpu_internal(range, to_fill);
        // SAFETY: after copy_to_cpu_internal returns, we know that to_fill was initialized for len elements
        unsafe { data.set_len(len) };
        data
    }

    pub fn copy_to_cpu_internal(&self, range: Range<usize>, dst: &mut [MaybeUninit<T>]) {
        if range.is_empty() {
            return;
        }

        let item_count = range.len();
        let source_offset = range.start as u64 * T::num_type().gpu_layout().size() as u64;

        assert!(dst.len() == item_count, "lengths must match");

        let mut staging_buffer = Self::empty_with_usage(item_count, BufferUsages::MAP_READ | BufferUsages::COPY_DST);
        let index = self.copy_to_internal(range, 0..item_count, &mut staging_buffer, false).unwrap();

        let slice = staging_buffer.rawr().slice(..);

        let notify = NotifySync::new();
        let notify_recv = notify.clone();

        slice.map_async(wgpu::MapMode::Read, move |_| notify.notify());
        WgpuContext::get().device.poll(wgpu::Maintain::WaitForSubmissionIndex(index));
        notify_recv.wait();

        // SAFETY: We just checked the length of slice, and copy_from_slice checks that the lengths of bytes match
        unsafe {
            let mapped_slice = slice.get_mapped_range();

            let (_, dst_bytes, _) = dst.align_to_mut::<MaybeUninit<u8>>();
            let (_, src_bytes, _) = mapped_slice.align_to::<MaybeUninit<u8>>();
            dst_bytes.copy_from_slice(src_bytes);
        }
        staging_buffer.rawr().unmap();
    }

    pub fn read(&self, dst: &mut [T]) {
        self.read_range(.., dst)
    }
    pub fn read_range(&self, range: impl RangeBounds<usize>, dst: &mut [T]) {
        let range = Self::check_bounds(range, self.len());
        assert!(range.len() == dst.len(), "length of dst slice must match");
        assert!(size_of::<T>() == size_of::<MaybeUninit<T>>());
        // SAFETY: We know T is same layout as MaybeUninit<T> and T is GpuNum so all bit patterns are valid
        // so any data written to dst will be valid
        let dst = unsafe { dst.align_to_mut::<MaybeUninit<T>>().1 };
        self.copy_to_cpu_internal(range, dst);
    }
}

impl<T: GpuNum> Clone for GpuVec<T> {
    fn clone(&self) -> Self {
        let mut new = Self::empty_with_usage(self.len(), self.usage());
        self.copy_to(&mut new);
        new
    }
}
impl<T: GpuNum> Debug for GpuVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const MAX_DISPLAY_LEN: usize = 32;
        if self.len() > MAX_DISPLAY_LEN {
            let values = self.to_cpu(..MAX_DISPLAY_LEN);
            let count = self.len() - MAX_DISPLAY_LEN;
            f.debug_list().entries(values).entry(&format_args!("... {count} more")).finish()
        } else {
            let values = self.to_cpu(..);
            f.debug_list().entries(values).finish()
        }
    }
}

#[derive(Clone)]
pub struct NotifySync {
    inner: Arc<(Mutex<bool>, Condvar)>,
}

impl NotifySync {
    pub fn new() -> Self {
        Self {
            inner: Arc::new((Mutex::new(false), Condvar::new())),
        }
    }

    pub fn notify(&self) {
        let mut notified = self.inner.0.lock().unwrap();
        *notified = true;
        self.inner.1.notify_one();
    }

    pub fn wait(&self) {
        let mut guard = self.inner.0.lock().unwrap();
        while !*guard {
            guard = self.inner.1.wait(guard).unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    #[test]
    fn test_copy_from_to_cpu() {
        let data = (0..1024).map(|v| v as f32).collect::<Vec<_>>();
        let gpu_vec = GpuVec::new_storage(&data);

        let start = Instant::now();
        let values = gpu_vec.to_cpu(..);
        println!("time: {:.03?}", start.elapsed());
        println!("values: {:?}", gpu_vec);
        assert_eq!(data, values);

        for i in 0..1024 {
            let data = (0..i).map(|v| -v as f32).collect::<Vec<_>>();
            let gpu_vec = GpuVec::new_storage(&data);

            let mut values = gpu_vec.to_cpu(..);
            assert_eq!(data, values);
            values.fill(0.0);
            gpu_vec.read_range(.., &mut values);
            assert_eq!(data, values);
        }
    }
}
