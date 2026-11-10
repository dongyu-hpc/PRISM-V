"""Re-export of the root ``tool.estimate_depth`` (optional dependency: depth models)."""

try:
    from tool.estimate_depth import estimate_depth  # noqa: F401
except Exception as _e:  # depth model weights are optional for the verification layer
    def estimate_depth(*args, **kwargs):
        raise RuntimeError(f"depth estimation unavailable: {_e}")
