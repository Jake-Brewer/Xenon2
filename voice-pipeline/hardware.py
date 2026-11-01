"""How many CPU workers the voice models should start.

Hyperthreads share a core. One worker per logical processor makes Silero,
Whisper's feature extractor, and Piper fight each other and the RWKV runtime.
Dementia and Sloth both speak through this one sidecar, so both get this count.
"""
from __future__ import annotations

import os


def logical_cpus() -> int:
    return max(1, os.cpu_count() or 1)


def physical_cpus() -> int:
    logical = logical_cpus()
    detected = _physical_cpus()
    if detected is None or detected < 1:
        return logical
    return min(detected, logical)


def _physical_cpus() -> int | None:
    if os.name == "nt":
        return _windows_physical_cores()
    siblings = "/sys/devices/system/cpu/cpu0/topology/thread_siblings_list"
    try:
        with open(siblings, encoding="utf-8") as handle:
            raw = handle.read().strip()
    except OSError:
        return None
    if not raw:
        return None
    # "0,1" or "0-1" is the sibling set for cpu0. One id means no hyperthreading.
    count = 0
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            count += int(end) - int(start) + 1
        else:
            count += 1
    if count <= 1:
        return logical_cpus()
    return max(1, logical_cpus() // count)


def _windows_physical_cores() -> int | None:
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetLogicalProcessorInformation.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    kernel32.GetLogicalProcessorInformation.restype = ctypes.c_int

    length = ctypes.c_uint32(0)
    kernel32.GetLogicalProcessorInformation(None, ctypes.byref(length))
    if length.value == 0:
        return None

    words = (length.value + 7) // 8
    raw = (ctypes.c_uint64 * words)()
    ok = kernel32.GetLogicalProcessorInformation(ctypes.cast(raw, ctypes.c_void_p), ctypes.byref(length))
    if not ok:
        return None

    data = ctypes.string_at(ctypes.addressof(raw), length.value)
    # SYSTEM_LOGICAL_PROCESSOR_INFORMATION is 32 bytes on x64.
    # Relationship is a u32 at offset 8. RelationProcessorCore == 0.
    stride = 32
    cores = 0
    off = 0
    while off + stride <= len(data):
        relationship = int.from_bytes(data[off + 8 : off + 12], "little")
        if relationship == 0:
            cores += 1
        off += stride
    return cores or None
