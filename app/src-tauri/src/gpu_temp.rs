//! Keeps the laptop's GPU from running hot during long generations. A background thread reads the
//! temperature with `nvidia-smi` every second; `throttle()`, called between tokens, paces
//! generation from `pace_from_c()` up (a short wait per token, longer the hotter it is) so the GPU
//! settles below `HOT_C`, and still pauses outright at `HOT_C` until it is back down to `COOL_C`.
//! Machines without `nvidia-smi` (no NVIDIA GPU) never start the thread, and `throttle()` then does
//! nothing.
//!
//! Why pace: on the RTX 2060 laptop the GPU went from 70 to 80 C in about 5 s of writing, and the
//! pauses down to 72 C took 13-16 s each, so a 300-token reply wrote for 11 s out of 40.

use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::{Duration, Instant};

pub const HOT_C: u32 = 80;
pub const COOL_C: u32 = 72;
/// From here up, each token waits `pace_ms_per_c()` per degree above `pace_from_c() - 1`, so the
/// GPU settles around a target temperature instead of sawtoothing up to the hard pause at `HOT_C`.
/// `XENON_PACE_FROM_C` and `XENON_PACE_MS_PER_C` override the defaults so the pace can be swept at
/// runtime (scripts/pace_sweep.py). Defaults from that sweep on the RTX 2060 laptop, 500-token
/// replies: from 76 C at 8 ms/C gave 19.0 tok/s sustained with no pauses (peak 77 C); 74 C at
/// 12 ms/C gave 15.6; pacing earlier or harder was slower still; no pacing gave 14.1 with a 13 s
/// pause at 80 C.
const DEFAULT_PACE_FROM_C: u32 = 76;
const DEFAULT_PACE_MS_PER_C: u64 = 8;
const POLL: Duration = Duration::from_secs(1);

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

/// The temperature (°C) at which per-token pacing begins. Clamped below `HOT_C`.
pub fn pace_from_c() -> u32 {
    env_u32("XENON_PACE_FROM_C", DEFAULT_PACE_FROM_C).min(HOT_C - 1)
}

/// Milliseconds added to each token's wait per degree above `pace_from_c() - 1`.
fn pace_ms_per_c() -> u64 {
    u64::from(env_u32("XENON_PACE_MS_PER_C", DEFAULT_PACE_MS_PER_C as u32))
}
/// A pause never lasts longer than this, so a stuck sensor can't freeze a reply.
const MAX_PAUSE: Duration = Duration::from_secs(120);

static TEMP_C: AtomicU32 = AtomicU32::new(0);
static STARTED: AtomicBool = AtomicBool::new(false);

/// Reads the GPU temperature once. `None` when there is no NVIDIA GPU or driver.
pub fn read() -> Option<u32> {
    let mut cmd = Command::new("nvidia-smi");
    cmd.args(["--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let out = cmd.output().ok()?;
    String::from_utf8_lossy(&out.stdout).lines().next()?.trim().parse().ok()
}

/// Starts the monitor thread once, if this machine has a readable NVIDIA GPU.
pub fn start_monitor() {
    if STARTED.swap(true, Ordering::SeqCst) {
        return;
    }
    let Some(first) = read() else {
        return;
    };
    TEMP_C.store(first, Ordering::Relaxed);
    std::thread::spawn(|| loop {
        std::thread::sleep(POLL);
        if let Some(t) = read() {
            TEMP_C.store(t, Ordering::Relaxed);
        }
    });
}

/// The last temperature the monitor saw, or 0 when it isn't running.
pub fn current() -> u32 {
    TEMP_C.load(Ordering::Relaxed)
}

/// The wait before the next token at `temp_c`: none when cool, a little more per degree as the
/// GPU warms, so generation slows to what the cooling can carry instead of stopping.
pub fn pace_for(temp_c: u32) -> Duration {
    let from = pace_from_c();
    if temp_c < from || temp_c >= HOT_C {
        return Duration::ZERO;
    }
    Duration::from_millis(pace_ms_per_c() * u64::from(temp_c + 1 - from))
}

/// Paces or pauses while the GPU is hot. Returns how long it waited.
pub fn throttle() -> Duration {
    let now = current();
    if now < HOT_C {
        let wait = pace_for(now);
        if !wait.is_zero() {
            std::thread::sleep(wait);
        }
        return wait;
    }
    let started = Instant::now();
    let hot = current();
    while current() > COOL_C && started.elapsed() < MAX_PAUSE {
        std::thread::sleep(Duration::from_millis(250));
    }
    let waited = started.elapsed();
    eprintln!("xenon2: paused {:.0}s to let the GPU cool ({hot}°C -> {}°C)", waited.as_secs_f32(), current());
    waited
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pacing_grows_with_heat_and_hands_over_to_the_pause_at_hot() {
        let from = pace_from_c();
        assert_eq!(pace_for(from - 1), Duration::ZERO);
        assert_eq!(pace_for(from), Duration::from_millis(pace_ms_per_c()));
        assert!(pace_for(79) > pace_for(76));
        assert_eq!(pace_for(HOT_C), Duration::ZERO, "at HOT_C throttle() pauses instead");
    }
}
