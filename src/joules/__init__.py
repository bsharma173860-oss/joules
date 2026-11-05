"""joules: measure the energy, cost and carbon of any command or local model.

As a library: joules.Session() measures named phases of your own program, joules.measure() one block, and
joules.connect() puts the machine on WattOps (Live, Tracking and the GPU leaderboard) from a notebook."""
__version__ = "0.8.2"


def __getattr__(name):  # the library API loads on first use, so the command line stays quick to start
    if name in ("Session", "measure", "Phase", "NoMeters", "PhaseError"):
        from . import api
        return getattr(api, name)
    if name in ("connect", "ConnectError"):
        from . import connection as _connect
        return getattr(_connect, name)
    raise AttributeError(f"module 'joules' has no attribute {name!r}")
