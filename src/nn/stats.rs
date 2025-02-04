use half::bf16;
use rayon::{iter::ParallelIterator, slice::ParallelSlice};

#[derive(Debug, Clone, Copy)]
pub struct Stats {
    count: usize,
    sum: f64,
    square_sum: f64,
    min: f64,
    max: f64,
}

impl Stats {
    pub const fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            square_sum: 0.0,
            min: f64::MAX,
            max: f64::MIN,
        }
    }
    pub const fn len(&self) -> usize {
        self.count
    }
    pub const fn mean(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum / self.count as f64
    }
    pub fn std(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let mean = self.mean();
        let std = ((self.square_sum / self.count as f64) - mean * mean).sqrt();
        std
    }
    pub const fn min(&self) -> f64 {
        self.min
    }
    pub const fn max(&self) -> f64 {
        self.max
    }
    pub fn append(&mut self, other: Stats) {
        self.count += other.count;
        self.sum += other.sum;
        self.square_sum += other.square_sum;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }
    pub fn print(&self, prec: usize) -> String {
        format!(
            "count: {}, mean: {:.prec$}, std: {:.prec$}, min: {:.prec$}, max: {:.prec$}",
            self.count,
            self.mean(),
            self.std(),
            self.min,
            self.max,
            prec = prec
        )
    }
    pub fn compute(data: &[impl Into<f64> + Send + Sync + Copy]) -> Self {
        #[derive(Clone, Copy)]
        struct Accumulator {
            sum: f64,
            square_sum: f64,
            min: f64,
            max: f64,
        }
        let acc = Accumulator {
            sum: 0.0,
            square_sum: 0.0,
            min: f64::MAX,
            max: f64::MIN,
        };
        let accs = data.par_chunks(512).map(|arr| {
            arr.iter().fold(acc, |mut acc, b| {
                let b = (*b).into();
                acc.sum += b;
                acc.square_sum += b * b;
                acc.min = acc.min.min(b);
                acc.max = acc.max.max(b);
                acc
            })
        });
        let fold = accs.reduce(
            || acc,
            |mut a, b| {
                a.sum += b.sum;
                a.square_sum += b.square_sum;
                a.min = a.min.min(b.min);
                a.max = a.max.max(b.max);
                a
            },
        );
        Self {
            count: data.len(),
            sum: fold.sum,
            square_sum: fold.square_sum,
            min: fold.min,
            max: fold.max,
        }
    }
}
