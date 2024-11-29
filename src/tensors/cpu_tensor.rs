use rayon::prelude::*;
use wide::f32x8;

#[derive(Debug)]
pub struct CpuTensor {
    data: Vec<f32>,
    shape: [usize; 2],
}

impl CpuTensor {
    pub fn new(data: &[impl AsRef<[f32]>]) -> Self {
        let x = data.len();
        let mut y = None;

        let continus = data.iter().flat_map(|val| {
            let array = val.as_ref();
            if let Some(size) = y {
                assert!(size == array.len(), "all arrays must be the same size");
            } else {
                y = Some(array.len());
            }
            array.iter().copied()
        });
        let data = continus.collect::<Vec<_>>();
        let shape = [x, y.unwrap()];

        Self { data, shape }
    }

    pub fn empty(shape: [usize; 2]) -> Self {
        assert!(shape[0] > 0 && shape[1] > 0, "shape must be greater than 0");
        let size = shape[0] * shape[1];
        Self { data: vec![0.0; size], shape }
    }
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    pub fn matrix_mul(&self, other: &CpuTensor) -> CpuTensor {
        assert!(
            self.shape[1] == other.shape[0],
            "width (shape[1]) of first tensor must be equal to height (shape[0]) of second tensor"
        );
        let mut out = Self::empty([self.shape[0], other.shape[1]]);
        matmul(&mut out.data, &self.data, &other.data, self.shape[0], other.shape[1]);
        out
    }
}

pub fn matmul(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, o: usize) {
    let n_simd = n / 8;

    xout.par_chunks_exact_mut(o).enumerate().for_each(|(j, elem)| {
        let xi = j * n;

        elem.par_chunks_exact_mut(4).enumerate().for_each(|(i, xout_elem)| {
            let new_i = i * 4;
            let ni0: usize = new_i * n;
            let ni1: usize = (new_i + 1) * n;
            let ni2: usize = (new_i + 2) * n;
            let ni3: usize = (new_i + 3) * n;

            xout_elem.iter_mut().for_each(|m| *m = 0.0);

            for j in 0..n_simd {
                let x_vec = f32x8::from(&x[xi + j * 8..xi + j * 8 + 8]);
                let w_vec0 = f32x8::from(&w[ni0 + j * 8..ni0 + j * 8 + 8]);
                let w_vec1 = f32x8::from(&w[ni1 + j * 8..ni1 + j * 8 + 8]);
                let w_vec2 = f32x8::from(&w[ni2 + j * 8..ni2 + j * 8 + 8]);
                let w_vec3 = f32x8::from(&w[ni3 + j * 8..ni3 + j * 8 + 8]);

                xout_elem[0] += (x_vec * w_vec0).reduce_add();
                xout_elem[1] += (x_vec * w_vec1).reduce_add();
                xout_elem[2] += (x_vec * w_vec2).reduce_add();
                xout_elem[3] += (x_vec * w_vec3).reduce_add();
            }
        });
    });
}
