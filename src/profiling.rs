use std::collections::HashSet;
use std::process::Stdio;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use comfy_table::{Cell, ContentArrangement, Table, presets::UTF8_FULL};
use sysinfo::{Pid, ProcessesToUpdate, System};

use crate::backend::Ct2PythonBackend;
use crate::config::AppConfig;
use crate::types::{MODEL_CT2_ALIAS, MODEL_HF_REPO, Transcript, TranscriptionRequest};

#[derive(Debug, Clone)]
pub struct ResourceSample {
    pub elapsed_seconds: f64,
    pub cpu_percent: f32,
    pub memory_mib: f64,
    pub virtual_memory_mib: f64,
    pub process_count: usize,
}

#[derive(Debug, Clone)]
pub struct ResourceSummary {
    pub avg_cpu_percent: f32,
    pub peak_cpu_percent: f32,
    pub avg_memory_mib: f64,
    pub peak_memory_mib: f64,
    pub avg_virtual_memory_mib: f64,
    pub peak_virtual_memory_mib: f64,
    pub peak_process_count: usize,
}

#[derive(Debug, Clone)]
pub struct ProfileReport {
    pub backend: String,
    pub model: String,
    pub elapsed_seconds: f64,
    pub audio_duration_seconds: f32,
    pub realtime_factor: Option<f64>,
    pub transcript: Transcript,
    pub samples: Vec<ResourceSample>,
    pub summary: ResourceSummary,
}

pub fn profile_request(
    config: &AppConfig,
    request: &TranscriptionRequest,
    sample_interval: Duration,
) -> Result<ProfileReport> {
    match request.backend {
        crate::types::BackendKind::Ct2Python => {
            profile_ct2_request(config, request, sample_interval)
        }
        crate::types::BackendKind::FpgaHybrid => {
            bail!("system profiling is not implemented for the fpga-hybrid backend yet")
        }
    }
}

fn profile_ct2_request(
    config: &AppConfig,
    request: &TranscriptionRequest,
    sample_interval: Duration,
) -> Result<ProfileReport> {
    let backend = Ct2PythonBackend::new(
        config.worker_launcher.clone(),
        config.worker_launcher_args.clone(),
        config.worker_script.clone(),
    );
    let mut command = backend.build_worker_command(request);
    command.stdout(Stdio::piped()).stderr(Stdio::piped());

    let started = Instant::now();
    let mut child = command
        .spawn()
        .with_context(|| format!("failed to spawn worker {}", config.worker_script.display()))?;
    let root_pid = Pid::from_u32(child.id());
    let mut system = System::new_all();
    let mut samples = Vec::new();

    loop {
        system.refresh_processes(ProcessesToUpdate::All, true);
        if let Some(sample) = collect_sample(&system, root_pid, started.elapsed().as_secs_f64()) {
            samples.push(sample);
        }

        if child.try_wait()?.is_some() {
            break;
        }

        thread::sleep(sample_interval);
    }

    let output = child.wait_with_output()?;
    let elapsed_seconds = started.elapsed().as_secs_f64();
    if !output.status.success() {
        bail!(
            "Python worker exited with status {}.\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let mut transcript: Transcript = serde_json::from_slice(&output.stdout).with_context(|| {
        format!(
            "failed to parse worker JSON output.\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    })?;
    transcript.model = MODEL_HF_REPO.to_owned();
    if transcript
        .notes
        .iter()
        .all(|note| !note.contains("Model alias:"))
    {
        transcript
            .notes
            .insert(1, format!("Model alias: {MODEL_CT2_ALIAS}"));
    }

    let realtime_factor = if transcript.audio_duration_seconds > 0.0 {
        Some(elapsed_seconds / f64::from(transcript.audio_duration_seconds))
    } else {
        None
    };

    let summary = summarize_samples(&samples);

    Ok(ProfileReport {
        backend: transcript.backend.clone(),
        model: transcript.model.clone(),
        elapsed_seconds,
        audio_duration_seconds: transcript.audio_duration_seconds,
        realtime_factor,
        transcript,
        samples,
        summary,
    })
}

fn collect_sample(system: &System, root_pid: Pid, elapsed_seconds: f64) -> Option<ResourceSample> {
    let mut visited = HashSet::new();
    let mut total_cpu = 0.0_f32;
    let mut total_memory = 0_u64;
    let mut total_virtual_memory = 0_u64;
    accumulate_process_tree(
        system,
        root_pid,
        &mut visited,
        &mut total_cpu,
        &mut total_memory,
        &mut total_virtual_memory,
    )?;

    Some(ResourceSample {
        elapsed_seconds,
        cpu_percent: total_cpu,
        memory_mib: bytes_to_mib(total_memory),
        virtual_memory_mib: bytes_to_mib(total_virtual_memory),
        process_count: visited.len(),
    })
}

fn accumulate_process_tree(
    system: &System,
    pid: Pid,
    visited: &mut HashSet<Pid>,
    total_cpu: &mut f32,
    total_memory: &mut u64,
    total_virtual_memory: &mut u64,
) -> Option<()> {
    if !visited.insert(pid) {
        return Some(());
    }

    let process = system.process(pid)?;
    *total_cpu += process.cpu_usage();
    *total_memory += process.memory();
    *total_virtual_memory += process.virtual_memory();
    let task_pids = process.tasks().cloned().unwrap_or_default();

    for (child_pid, child_process) in system.processes() {
        if child_process.parent() == Some(pid) && !task_pids.contains(child_pid) {
            accumulate_process_tree(
                system,
                *child_pid,
                visited,
                total_cpu,
                total_memory,
                total_virtual_memory,
            );
        }
    }

    Some(())
}

fn summarize_samples(samples: &[ResourceSample]) -> ResourceSummary {
    if samples.is_empty() {
        return ResourceSummary {
            avg_cpu_percent: 0.0,
            peak_cpu_percent: 0.0,
            avg_memory_mib: 0.0,
            peak_memory_mib: 0.0,
            avg_virtual_memory_mib: 0.0,
            peak_virtual_memory_mib: 0.0,
            peak_process_count: 0,
        };
    }

    let len = samples.len() as f32;
    let avg_cpu_percent = samples.iter().map(|sample| sample.cpu_percent).sum::<f32>() / len;
    let peak_cpu_percent = samples
        .iter()
        .map(|sample| sample.cpu_percent)
        .fold(0.0_f32, f32::max);
    let avg_memory_mib =
        samples.iter().map(|sample| sample.memory_mib).sum::<f64>() / samples.len() as f64;
    let peak_memory_mib = samples
        .iter()
        .map(|sample| sample.memory_mib)
        .fold(0.0_f64, f64::max);
    let avg_virtual_memory_mib = samples
        .iter()
        .map(|sample| sample.virtual_memory_mib)
        .sum::<f64>()
        / samples.len() as f64;
    let peak_virtual_memory_mib = samples
        .iter()
        .map(|sample| sample.virtual_memory_mib)
        .fold(0.0_f64, f64::max);
    let peak_process_count = samples
        .iter()
        .map(|sample| sample.process_count)
        .fold(0_usize, usize::max);

    ResourceSummary {
        avg_cpu_percent,
        peak_cpu_percent,
        avg_memory_mib,
        peak_memory_mib,
        avg_virtual_memory_mib,
        peak_virtual_memory_mib,
        peak_process_count,
    }
}

fn bytes_to_mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

pub fn render_summary_table(report: &ProfileReport) -> String {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            "backend",
            "audio_s",
            "elapsed_s",
            "rtf",
            "avg_cpu_%",
            "peak_cpu_%",
            "avg_ram_mib",
            "peak_ram_mib",
            "peak_vram_mib",
            "peak_procs",
        ])
        .add_row(vec![
            Cell::new(&report.backend),
            Cell::new(format!("{:.3}", report.audio_duration_seconds)),
            Cell::new(format!("{:.3}", report.elapsed_seconds)),
            Cell::new(
                report
                    .realtime_factor
                    .map(|value| format!("{value:.3}x"))
                    .unwrap_or_else(|| "n/a".to_owned()),
            ),
            Cell::new(format!("{:.1}", report.summary.avg_cpu_percent)),
            Cell::new(format!("{:.1}", report.summary.peak_cpu_percent)),
            Cell::new(format!("{:.1}", report.summary.avg_memory_mib)),
            Cell::new(format!("{:.1}", report.summary.peak_memory_mib)),
            Cell::new(format!("{:.1}", report.summary.peak_virtual_memory_mib)),
            Cell::new(report.summary.peak_process_count),
        ]);
    table.to_string()
}

pub fn render_samples_table(report: &ProfileReport) -> String {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["t_s", "cpu_%", "ram_mib", "virt_mib", "procs"]);

    for sample in &report.samples {
        table.add_row(vec![
            Cell::new(format!("{:.3}", sample.elapsed_seconds)),
            Cell::new(format!("{:.1}", sample.cpu_percent)),
            Cell::new(format!("{:.1}", sample.memory_mib)),
            Cell::new(format!("{:.1}", sample.virtual_memory_mib)),
            Cell::new(sample.process_count),
        ]);
    }

    table.to_string()
}
