#![allow(unused)]
use std::{
    fs::File,
    path::{Path, PathBuf},
};

mod llama;
mod nn;
mod tensors;

use bytes::Bytes;
use safetensors::{tensor, SafeTensors, View};
use tensors::GpuTensor;

fn memory_map_file_unchecked(path: &Path) -> Bytes {
    let file = File::open(path).unwrap();
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
    Bytes::from_owner(mmap) // doesn't copy any memory
}

#[cfg(target_os = "linux")]
const MODELS_DIR: &str = "/mnt/d/MyFolder/Projekty/rust_lm.rs/models";
#[cfg(target_os = "windows")]
const MODELS_DIR: &str = "D:/MyFolder/Projekty/rust_lm.rs/models";

const TEST_FILES_PATH: &str = constcat::concat!(MODELS_DIR, "/LLama-3-8b-Uncensored");
const TEST_FILES_PATH2: &str = constcat::concat!(MODELS_DIR, "/CodeQwen");
const TEST_FILES_PATH3: &str = constcat::concat!(MODELS_DIR, "/whisper");

fn tensor_files_in(dir: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|entry| {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_file() && path.extension().is_some_and(|ext| ext == "safetensors") {
                Some(path)
            } else {
                None
            }
        })
        .collect()
}
// todo wgpu tensor:
// https://github.com/kurtschelfthout/tensorken/blob/v0.2/src/raw_tensor_wgpu.rs
fn scan_safetensors() {
    let mut shapes = Vec::new();
    for path in tensor_files_in(Path::new(TEST_FILES_PATH)) {
        println!("Path {}", path.display());
        let mmap = memory_map_file_unchecked(&path);
        let tensors = SafeTensors::deserialize(&mmap).unwrap();
        let tens = tensors.tensors();
        println!("{} tensors", tens.len());
        for (name, tensor) in tens {
            shapes.push((tensor.shape().to_vec(), tensor.dtype()));
            if !name.contains("layers.0") {
                continue;
            }
            println!("{}: dtype:{:?}, shape:{:?}", name, tensor.dtype(), tensor.shape());
        }
    }
    let diff_shapes = shapes.into_iter().collect::<std::collections::BTreeSet<_>>();
    println!("unique shapes {:?}", diff_shapes);
}
fn main() {
    let data1 = (0..512).map(|v| v as f32).collect::<Vec<_>>();
    let data2 = (0..512).map(|v| v as f32).collect::<Vec<_>>();

    // loop {
    //     std::hint::spin_loop();
    // }

    //let result = GpuTensor::matrix_mul_native(&data1, &data2, indexes).await;
    //println!("result: {result:?}");
    scan_safetensors();
}
// model.layers.0.input_layernorm.weight: dtype:F16, shape:[4096]
// model.layers.0.self_attn.k_proj.weight: dtype:F16, shape:[1024, 4096]
// model.layers.0.self_attn.v_proj.weight: dtype:F16, shape:[1024, 4096]
// model.layers.0.self_attn.q_proj.weight: dtype:F16, shape:[4096, 4096]

// model.layers.0.self_attn.o_proj.weight: dtype:F16, shape:[4096, 4096]

// model.layers.0.mlp.down_proj.weight: dtype:F16, shape:[4096, 14336]
// model.layers.0.mlp.gate_proj.weight: dtype:F16, shape:[14336, 4096]
// model.layers.0.mlp.up_proj.weight: dtype:F16, shape:[14336, 4096]
// model.layers.0.post_attention_layernorm.weight: dtype:F16, shape:[4096]
