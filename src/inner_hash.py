from dataclasses import dataclass

import numpy as np
import torch


def xor_reduction(inputs: torch.Tensor) -> np.uint32:
    """
    XOR all uint32 elements in the input tensor.
    Returns a single uint32 value.
    """
    assert inputs.dtype == torch.int32, f"Expected int32, got {inputs.dtype}"
    arr = inputs.flatten().numpy().view(np.uint32)
    return np.bitwise_xor.reduce(arr)


@dataclass
class InnerHashResult:
    hash: np.uint32
    index: tuple[int, int]


def hash_tile(
    tensor: torch.Tensor,
    index: tuple[int, int] | None = None,
) -> InnerHashResult:
    if tensor.dtype != torch.int32:
        raise ValueError(f"Tensor dtype must be int32, got {tensor.dtype}")

    final_hash = xor_reduction(tensor)

    return InnerHashResult(hash=final_hash, index=index if index is not None else (0, 0))


class InnerHasher:
    def __init__(self, tile_h: int, tile_w: int):
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.num_tiles_hashed = 0

    def hash_tensor(self, tensor: torch.Tensor) -> list[InnerHashResult]:
        if tensor.dtype != torch.int32:
            raise ValueError(f"Tensor dtype must be int32, got {tensor.dtype}")

        if tensor.shape[0] < self.tile_h or tensor.shape[1] < self.tile_w:
            raise ValueError(
                f"Tensor must have shape of at least ({self.tile_h}, {self.tile_w}), got {tensor.shape}"
            )

        num_tiles_h = tensor.shape[0] // self.tile_h
        num_tiles_w = tensor.shape[1] // self.tile_w

        self.num_tiles_hashed += num_tiles_h * num_tiles_w

        hashes = []

        for i in range(num_tiles_h):
            for j in range(num_tiles_w):
                tile = tensor[
                    i * self.tile_h : (i + 1) * self.tile_h,
                    j * self.tile_w : (j + 1) * self.tile_w,
                ]
                hashes.append(hash_tile(tile, (i, j)))

        return hashes
