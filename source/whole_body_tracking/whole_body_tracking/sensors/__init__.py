from __future__ import annotations

import inspect
import warnings
from typing import Literal

from .gpu_dob_contact_sensor import GpuDobContactSensor, GpuDobContactSensorData

DobContactBackend = Literal["gpu", "pinocchio", "auto"]

_PINOCCHIO_IMPORT_ERROR: ModuleNotFoundError | None = None
_PINOCCHIO_AVAILABLE = True

try:
    from .dob_contact_sensor import DobContactSensor, DobContactSensorData
except ModuleNotFoundError as exc:
    if exc.name != "pinocchio":
        raise
    _PINOCCHIO_AVAILABLE = False
    _PINOCCHIO_IMPORT_ERROR = exc
    DobContactSensorData = GpuDobContactSensorData

    class DobContactSensor(GpuDobContactSensor):  # type: ignore[no-redef]
        """Compatibility fallback used when the optional pinocchio package is absent."""

        def __init__(self, *args, **kwargs) -> None:
            warnings.warn(
                "pinocchio is not installed; DobContactSensor is falling back to "
                "GpuDobContactSensor. Use create_dob_contact_sensor(..., backend='gpu') "
                "to request this explicitly.",
                RuntimeWarning,
                stacklevel=2,
            )
            kwargs.pop("urdf_path", None)
            kwargs.pop("num_workers", None)
            super().__init__(*args, **kwargs)


def is_pinocchio_dob_available() -> bool:
    """Return whether the Pinocchio-backed DOB contact sensor can be constructed."""
    return _PINOCCHIO_AVAILABLE


def _warn_pinocchio_fallback(requested_backend: str) -> None:
    detail = f" ({_PINOCCHIO_IMPORT_ERROR})" if _PINOCCHIO_IMPORT_ERROR is not None else ""
    warnings.warn(
        f"DOB contact backend '{requested_backend}' requested, but pinocchio is not installed{detail}; "
        "using GPU/PhysX backend instead.",
        RuntimeWarning,
        stacklevel=3,
    )


def _resolve_dob_contact_sensor_class(
    backend: str = "gpu",
    *,
    fallback_to_gpu: bool = True,
) -> tuple[type, str]:
    backend = backend.lower()
    if backend == "gpu":
        return GpuDobContactSensor, "gpu"
    if backend == "pinocchio":
        if _PINOCCHIO_AVAILABLE:
            return DobContactSensor, "pinocchio"
        if fallback_to_gpu:
            _warn_pinocchio_fallback(backend)
            return GpuDobContactSensor, "gpu"
        raise RuntimeError(
            "Pinocchio DOB contact backend was requested, but the optional "
            "'pinocchio' package is not installed."
        )
    if backend == "auto":
        if _PINOCCHIO_AVAILABLE:
            return DobContactSensor, "pinocchio"
        _warn_pinocchio_fallback(backend)
        return GpuDobContactSensor, "gpu"
    raise ValueError("DOB contact backend must be one of: 'gpu', 'pinocchio', or 'auto'.")


def create_dob_contact_sensor(
    env,
    backend: str = "gpu",
    *,
    fallback_to_gpu: bool = True,
    **kwargs,
):
    """Create a DOB contact sensor with an optional Pinocchio-to-GPU fallback.

    ``backend='gpu'`` is the fast PhysX/PyTorch path. ``backend='pinocchio'``
    requests the original CPU Pinocchio implementation. ``backend='auto'`` uses
    Pinocchio when available and otherwise falls back to GPU.
    """
    sensor_cls, resolved_backend = _resolve_dob_contact_sensor_class(
        backend, fallback_to_gpu=fallback_to_gpu
    )
    params = inspect.signature(sensor_cls.__init__).parameters
    accepted = set(params) - {"self", "env"}
    sensor_kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    sensor = sensor_cls(env, **sensor_kwargs)
    sensor.requested_dob_backend = backend
    sensor.dob_backend = resolved_backend
    return sensor


def get_or_create_dob_contact_sensor(
    env,
    backend: str = "gpu",
    *,
    update: bool = True,
    **kwargs,
):
    """Return a cached DOB contact sensor attached to the unwrapped env."""
    unwrapped = env.unwrapped
    sensors = getattr(unwrapped, "_dob_contact_sensors", None)
    if sensors is None:
        sensors = {}
        unwrapped._dob_contact_sensors = sensors

    key = backend.lower()
    sensor = sensors.get(key)
    if sensor is None:
        sensor = create_dob_contact_sensor(unwrapped, backend=backend, **kwargs)
        sensors[key] = sensor
    if update:
        sensor.update()
    return sensor


__all__ = [
    "DobContactBackend",
    "DobContactSensor",
    "DobContactSensorData",
    "GpuDobContactSensor",
    "GpuDobContactSensorData",
    "create_dob_contact_sensor",
    "get_or_create_dob_contact_sensor",
    "is_pinocchio_dob_available",
]
