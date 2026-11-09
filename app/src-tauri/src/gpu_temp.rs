//! Keeps the laptop's GPU from running hot during long generations. A background thread reads the
//! temperature with `nvidia-smi` every few seconds; `throttle()`, called between tokens, pauses
//! generation once the GPU reaches `HOT_C` until it is back down to `COOL_C`. Machines without
//! `nvidia-smi` (no NVIDIA GPU) never start the thread, and `throttle()` then does nothing.

use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::{Duration, Instant};

pub const HOT_C: u32 = 80;
pub const COOL_C: u32 = 72;
const POLL: Duration = Duration::from_secs(3);
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

/// Waits while the GPU is hot. Returns how long it waited, so callers can log or report it.
pub fn throttle() -> Duration {
    if current() < HOT_C {
        return Duration::ZERO;
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
