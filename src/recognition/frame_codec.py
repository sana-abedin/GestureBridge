
"""
Package 16 JPEG frames for the Hugging Face ISL API.
"""

import struct

MAGIC = b"ISLF"
_HEADER = struct.Struct("<4sH")


def pack_frames(jpegs: list[bytes]) -> bytes:
    if not jpegs:
        raise ValueError("No frames to pack")

    if len(jpegs) != 16:
        raise ValueError(
            f"Expected 16 frames, got {len(jpegs)}"
        )

    if any(
        len(frame) == 0 or len(frame) > 500_000
        for frame in jpegs
    ):
        raise ValueError("Invalid JPEG frame length")

    lengths = struct.pack(
        f"<{len(jpegs)}I",
        *map(len, jpegs)
    )

    return b"".join([
        _HEADER.pack(MAGIC, len(jpegs)),
        lengths,
        *jpegs
    ])
