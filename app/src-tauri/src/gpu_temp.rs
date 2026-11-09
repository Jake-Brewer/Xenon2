//! Keeps the laptop's GPU from running hot during long generations. A background thread reads the
//! temperature with `nvidia-smi` every second; `throttle()`, called between tokens, paces
//! generation from `PACE_FROM_C` up (a short wait per token, longer the hotter it is) so the GPU
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
/// From here up, each token waits `PACE_MS_PER_C` per degree above `PACE_FROM_C - 1`.
pub const PACE_FROM_C: u32 = 74;
const PACE_MS_PER_C: u64 = 12;
const POLL: Duration = Duration::from_secs(1);
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
    if temp_c < PACE_FROM_C || temp_c >= HOT_C {
        return Duration::ZERO;
    }
    Duration::from_millis(PACE_MS_PER_C * u64::from(temp_c + 1 - PACE_FROM_C))
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
        assert_eq!(pace_for(70), Duration::ZERO);
        assert_eq!(pace_for(PACE_FROM_C), Duration::from_millis(PACE_MS_PER_C));
        assert!(pace_for(79) > pace_for(76));
        assert_eq!(pace_for(HOT_C), Duration::ZERO, "at HOT_C throttle() pauses instead");
    }
}
