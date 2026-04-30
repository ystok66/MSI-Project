"""Experiment entry points for risky_maze."""


def run_fixed_experiment(*args, **kwargs):
    from .run_fixed_maze import run_fixed_experiment as _impl

    return _impl(*args, **kwargs)


__all__ = ["run_fixed_experiment"]
