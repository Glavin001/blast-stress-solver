use std::{
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
};

use anyhow::{Context, Result};

pub struct GpuTelemetry {
    child: Option<Child>,
    csv_path: PathBuf,
    summary_path: PathBuf,
}

impl GpuTelemetry {
    pub fn start(csv_path: PathBuf, summary_path: PathBuf) -> Result<Self> {
        let mut header = File::create(&csv_path)
            .with_context(|| format!("create telemetry {}", csv_path.display()))?;
        writeln!(
            header,
            "timestamp,gpu_util_percent,memory_used_mib,power_w,temperature_c,sm_clock_mhz"
        )?;
        drop(header);

        let output = OpenOptions::new()
            .append(true)
            .open(&csv_path)
            .with_context(|| format!("append telemetry {}", csv_path.display()))?;
        let child = match Command::new("nvidia-smi")
            .args([
                "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm",
                "--format=csv,noheader,nounits",
                "-lms",
                "200",
            ])
            .stdout(Stdio::from(output))
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(child) => Some(child),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                eprintln!("warning: nvidia-smi not found; GPU telemetry disabled");
                None
            }
            Err(error) => return Err(error).context("start nvidia-smi telemetry"),
        };
        println!("gpu_log={}", csv_path.display());
        Ok(Self {
            child,
            csv_path,
            summary_path,
        })
    }

    pub fn stop(mut self) -> Result<()> {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        summarize(&self.csv_path, &self.summary_path)?;
        println!("gpu_summary={}", self.summary_path.display());
        Ok(())
    }
}

impl Drop for GpuTelemetry {
    fn drop(&mut self) {
        if let Some(child) = &mut self.child {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn summarize(csv_path: &Path, summary_path: &Path) -> Result<()> {
    let file =
        File::open(csv_path).with_context(|| format!("read telemetry {}", csv_path.display()))?;
    let mut count = 0_u64;
    let mut sums = [0.0_f64; 5];
    let mut maxima = [0.0_f64; 5];

    for line in BufReader::new(file).lines().skip(1) {
        let line = line?;
        let fields: Vec<_> = line.split(',').map(str::trim).collect();
        if fields.len() != 6 {
            continue;
        }
        let mut values = [0.0_f64; 5];
        let mut valid = true;
        for (index, field) in fields[1..].iter().enumerate() {
            match field.parse::<f64>() {
                Ok(value) => values[index] = value,
                Err(_) => {
                    valid = false;
                    break;
                }
            }
        }
        if !valid {
            continue;
        }
        count += 1;
        for index in 0..5 {
            sums[index] += values[index];
            maxima[index] = maxima[index].max(values[index]);
        }
    }

    let mut summary = File::create(summary_path)
        .with_context(|| format!("write telemetry summary {}", summary_path.display()))?;
    if count == 0 {
        writeln!(summary, "No GPU samples captured.")?;
        return Ok(());
    }
    writeln!(summary, "samples={count}")?;
    writeln!(summary, "gpu_util_avg={:.1}%", sums[0] / count as f64)?;
    writeln!(summary, "gpu_util_max={:.1}%", maxima[0])?;
    writeln!(summary, "memory_avg={:.0} MiB", sums[1] / count as f64)?;
    writeln!(summary, "memory_max={:.0} MiB", maxima[1])?;
    writeln!(summary, "power_avg={:.1} W", sums[2] / count as f64)?;
    writeln!(summary, "power_max={:.1} W", maxima[2])?;
    writeln!(summary, "temperature_avg={:.1} C", sums[3] / count as f64)?;
    writeln!(summary, "temperature_max={:.0} C", maxima[3])?;
    writeln!(summary, "sm_clock_avg={:.0} MHz", sums[4] / count as f64)?;
    writeln!(summary, "sm_clock_max={:.0} MHz", maxima[4])?;
    Ok(())
}
