"""Keeps the Windows input level of the active mic in a useful range.

WebRTC's gain control adjusts the signal after it is captured, which cannot undo clipping or a
level set far too low. This nudges the real Windows level from how loud finished utterances
were. It affects every app using that mic, so it moves in small steps inside fixed limits, and
it ignores anything captured while Xenon was speaking.
"""
from __future__ import annotations

import sys
from typing import Optional

QUIET_PEAK = 0.12
LOUD_PEAK = 0.70
RAISE_DB = 2.0
LOWER_DB = 4.0
MIN_DB = -24.0  # some headset mics top out at 0 dB, so the floor must sit below that
MAX_DB = 24.0


def _log(msg: str) -> None:
    print(f"[audio] {msg}", file=sys.stderr, flush=True)


def _endpoint(device_name: str):
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    for device in AudioUtilities.GetAllDevices():
        if device.FriendlyName == device_name:
            interface = device._dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            return interface.QueryInterface(IAudioEndpointVolume)
    return None


def next_level(current_db: float, peak: float, device_min: float = MIN_DB, device_max: float = MAX_DB) -> Optional[float]:
    """The level to move to for an utterance with this peak, or None to stay put. Each device has
    its own range, and asking for a level outside it fails."""
    if peak > LOUD_PEAK:
        target = current_db - LOWER_DB
    elif peak < QUIET_PEAK:
        target = current_db + RAISE_DB
    else:
        return None
    target = min(MAX_DB, device_max, max(MIN_DB, device_min, target))
    return None if abs(target - current_db) < 0.1 else target


def adjust_for_utterance(device_name: str, peak: float) -> None:
    """Never raises. A failed adjustment only means the level stays where it is."""
    if sys.platform != "win32" or peak <= 0.0:
        return
    try:
        endpoint = _endpoint(device_name)
        if endpoint is None:
            return
        current = float(endpoint.GetMasterVolumeLevel())
        device_min, device_max, _step = endpoint.GetVolumeRange()
        target = next_level(current, peak, float(device_min), float(device_max))
        if target is not None:
            endpoint.SetMasterVolumeLevel(target, None)
            _log(f"mic level {current:.1f} dB -> {target:.1f} dB (speech peak {peak:.3f})")
    except Exception as exc:  # noqa: BLE001 -- level control must never break listening
        _log(f"mic level adjust failed: {exc}")
