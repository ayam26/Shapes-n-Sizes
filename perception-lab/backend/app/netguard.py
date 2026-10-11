"""Application-level outbound network guard.

When enabled, every outbound socket connect from this Python process must target a
loopback address. Incoming connections (the SSH tunnel to uvicorn) are unaffected,
and host-wide networking is untouched. Limits: it does not cover subprocesses or
other processes (e.g. the Ollama server itself), only this backend."""
import ipaddress
import socket

_original_connect = socket.socket.connect
_enabled = False


def _is_loopback(address) -> bool:
    if isinstance(address, (str, bytes)):  # AF_UNIX path
        return True
    host = address[0]
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False  # unresolved hostname: refuse rather than resolve externally


def _guarded_connect(self, address):
    if not _is_loopback(address):
        raise ConnectionRefusedError(f"outbound connection to {address!r} blocked by PL_OFFLINE_GUARD")
    return _original_connect(self, address)


def enable() -> None:
    global _enabled
    socket.socket.connect = _guarded_connect
    _enabled = True


def disable() -> None:
    global _enabled
    socket.socket.connect = _original_connect
    _enabled = False


def enabled() -> bool:
    return _enabled
