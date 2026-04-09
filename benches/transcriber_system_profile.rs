use std::hint::black_box;
use std::path::PathBuf;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use fpga_whisper::config::AppConfig;
use fpga_whisper::profiling::{profile_request, render_summary_table};
use fpga_whisper::types::{BackendKind, PartitionPreset, TranscriptionRequest};

fn criterion_benchmark(c: &mut Criterion) {
    let config = AppConfig::default();
    let request = TranscriptionRequest {
        audio_path: PathBuf::from("samples/jfk.flac"),
        backend: BackendKind::Ct2Python,
        partition: PartitionPreset::CpuOnly,
        initial_prompt: None,
    };

    let mut group = c.benchmark_group("transcriber_system_profile");
    group.sample_size(10);
    group.bench_function("ct2_python_jfk", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            let mut last_report = None;

            for _ in 0..iters {
                let started = std::time::Instant::now();
                let report = profile_request(&config, &request, Duration::from_millis(250))
                    .expect("profile request should succeed");
                total += started.elapsed();
                last_report = Some(report);
            }

            if std::env::var_os("FPGA_WHISPER_PRINT_PROFILE").is_some() {
                if let Some(report) = last_report.as_ref() {
                    println!("\n{}", render_summary_table(report));
                }
            }

            if let Some(report) = last_report {
                black_box(report);
            }

            total
        });
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
