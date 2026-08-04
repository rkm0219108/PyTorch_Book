"""
Routines for setting CUDA specific properties.
"""
from __future__ import annotations
import typing
__all__: list[str] = ['get_device', 'get_num_devices', 'set_device']
def get_device() -> int:
    """
    Get the active CUDA device.
    """
def get_num_devices() -> int:
    """
    Find out how many CUDA devices are available.
    """
def set_device(device_id: typing.SupportsInt | typing.SupportsIndex) -> None:
    """
    Set the active CUDA device.  It is required that 0 <= device_id < get_num_devices().
    """
