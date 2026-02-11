"""Shared CLI argument parsing utilities.

Provides ``parse_unknown_args()`` for converting leftover argparse unknowns
into a properly typed ``dict`` for passing as ``**kwargs`` to framework APIs
(e.g. ``model.train(**kwargs)``).
"""


def parse_unknown_args(unknown):
    """Parse a list of leftover CLI tokens into a typed ``dict``.

    Converts ``argparse.parse_known_args()`` unknowns such as
    ``["--lr0", "0.01", "--freeze", "10", "--nms", "true"]`` into
    ``{"lr0": 0.01, "freeze": 10, "nms": True}``.

    Type conversion rules (applied in order):
        - ``"true"`` / ``"false"`` (case-insensitive) → ``bool``
        - ``"none"`` (case-insensitive) → ``None``
        - Integer string → ``int``
        - Float string → ``float``
        - Comma-separated values → ``tuple`` of converted elements
        - Everything else → ``str``

    Flags without a following value (e.g. ``--verbose``) are set to ``True``.

    Args:
        unknown: List of strings from ``argparse.parse_known_args()``.

    Returns:
        dict: Typed key-value pairs ready for ``**kwargs`` usage.

    Examples:
        >>> parse_unknown_args(["--lr0", "0.01", "--freeze", "10", "--verbose"])
        {'lr0': 0.01, 'freeze': 10, 'verbose': True}
        >>> parse_unknown_args(["--imgsz", "640,640", "--nms", "false"])
        {'imgsz': (640, 640), 'nms': False}
    """
    kwargs = {}
    i = 0
    while i < len(unknown):
        token = unknown[i]
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            # Check if next token is a value (not another flag)
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                kwargs[key] = _convert_value(unknown[i + 1])
                i += 2
            else:
                # Bare flag → True
                kwargs[key] = True
                i += 1
        else:
            # Skip non-flag tokens (shouldn't happen with argparse leftovers)
            i += 1
    return kwargs


def _convert_value(value):
    """Convert a string CLI value to the most appropriate Python type.

    Args:
        value: Raw string value from the command line.

    Returns:
        The converted value (bool, None, int, float, tuple, or str).
    """
    low = value.lower()

    # Booleans
    if low == "true":
        return True
    if low == "false":
        return False

    # None
    if low == "none":
        return None

    # Comma-separated → tuple of converted elements
    if "," in value:
        return tuple(_convert_value(v.strip()) for v in value.split(","))

    # Integer
    try:
        return int(value)
    except ValueError:
        pass

    # Float
    try:
        return float(value)
    except ValueError:
        pass

    # Fallback: string
    return value
