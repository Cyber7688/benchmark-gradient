from math import ceil

import numpy as np
import torch
from blake3 import blake3
from miner_utils import get_logger

logger = get_logger(__name__)


def is_power_of_two(n: int) -> bool:
    return n != 0 and (n & (n - 1)) == 0


def num_to_bytes(n: int) -> bytes:
    byte_length = (n.bit_length() + 7) // 8
    return n.to_bytes(byte_length, "big")


def np_mul_hi_u32(a: np.uint32, b: np.uint32):
    prod64 = a.astype(np.uint64) * b.astype(np.uint64)
    return (prod64 >> np.uint64(32)).astype(np.uint32)


class NoiseGenerator:
    """A class for generating noise for a given tensor."""

    def __init__(self, noise_rank: int = 128, noise_range: int = 128):
        self.noise_rank = noise_rank

        if not is_power_of_two(noise_range):
            raise ValueError("noise_range must be a power of two")

        if not is_power_of_two(noise_rank):
            raise ValueError("noise_rank must be a power of two")

        if noise_range > 128:
            raise ValueError("noise_range must fit in uint7")

        if noise_rank % blake3.digest_size != 0:
            raise ValueError(f"noise_rank must be divisible by {blake3.digest_size=}")

        logger.info(
            f"Initialized NoiseGenerator with noise_rank={noise_rank}, noise_range={noise_range}"
        )

        # noise_range is a power of two, so we can use a mask to get the lower bits
        idxs_per_col = 2
        _noise_range = noise_range // idxs_per_col
        self.zero_point_translation = _noise_range // 2
        self.range_mask = _noise_range - 1
        self.rank_mask = self.noise_rank - 1

    def generate_noise_metrices(
        self,
        key_A: bytes,
        key_B: bytes,
        A_rows: int,
        common_dim: int,
        B_cols: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if A_rows < self.noise_rank:
            raise ValueError(f"A_rows must be greater than or equal to noise_rank, got {A_rows}")
        if common_dim < self.noise_rank:
            raise ValueError(
                f"common_dim must be greater than or equal to noise_rank, got {common_dim}"
            )
        if B_cols < self.noise_rank:
            raise ValueError(f"B_cols must be greater than or equal to noise_rank, got {B_cols}")

        seed_A = (
            torch.frombuffer(bytearray(b"A_tensor" + b"\x00" * 24), dtype=torch.uint8)
            .numpy()
            .tobytes()
        )
        seed_B = (
            torch.frombuffer(bytearray(b"B_tensor" + b"\x00" * 24), dtype=torch.uint8)
            .numpy()
            .tobytes()
        )

        A_L = self.__generate_uniform_random_matrix(seed_A, key_A, A_rows)
        A_R = self.__generate_permutation_matrix(
            seed_A,
            key_A,
            self.noise_rank,
            common_dim,
            assign_columns=True,
        )
        B_L = self.__generate_permutation_matrix(
            seed_B,
            key_B,
            common_dim,
            self.noise_rank,
            assign_columns=False,
        )
        B_R = self.__generate_uniform_random_matrix(seed_B, key_B, B_cols).T

        return A_L, A_R, B_L, B_R

    def __get_random_hash(self, index: int, seed: bytes, key: bytes, prepend_index: int) -> bytes:
        message_prepend = np.zeros(8, dtype=np.int32)
        message_prepend[prepend_index] = 1 + index
        message_bytes = message_prepend.tobytes() + seed
        return blake3(message_bytes, key=key).digest()

    def __generate_uniform_random_matrix(
        self,
        seed: bytes,
        key: bytes,
        rows: int,
    ) -> torch.Tensor:
        """Generate a uniform random matrix."""

        cols = self.noise_rank
        assert len(seed) == 32, "seed must be 32 bytes"
        assert len(key) == 32, "key must be 32 bytes"
        noise_matrix = torch.zeros(rows, cols, dtype=torch.int8)

        draws = int(ceil(rows * cols / blake3.digest_size))

        # draw all required hashes
        random_bytes = b"".join(self.__get_random_hash(i, seed, key, 0) for i in range(draws))

        # Convert to tensor and apply operations in one go
        random_tensor = torch.frombuffer(bytearray(random_bytes), dtype=torch.uint8)[: rows * cols]

        # Vectorized mask, translation, and reshape
        noise_matrix = (
            ((random_tensor & self.range_mask) - self.zero_point_translation)
            .to(torch.int8)
            .view(rows, cols)
        )

        return noise_matrix

    def __generate_permutation_matrix(
        self,
        seed: bytes,
        key: bytes,
        rows: int,
        cols: int,
        assign_columns: bool,
    ) -> torch.Tensor:
        """Generate a permutation matrix."""
        assert (rows == self.noise_rank) or (cols == self.noise_rank), (
            "rows or cols must be equal to noise_rank"
        )
        assert len(seed) == 32, "seed must be 32 bytes"
        assert len(key) == 32, "key must be 32 bytes"
        noise_matrix = torch.zeros(rows, cols, dtype=torch.int8)

        if assign_columns:
            required_lines = cols
            assert rows == self.noise_rank, "rows must be equal to noise_rank"
        else:
            required_lines = rows
            assert cols == self.noise_rank, "cols must be equal to noise_rank"

        bytes_per_lines = 4
        draws = int(ceil(required_lines * bytes_per_lines / blake3.digest_size))

        # We draw 4 bytes for each matrix element
        for i in range(draws):
            random_hash = self.__get_random_hash(i, seed, key, 1)
            random_uint32_array = np.frombuffer(
                random_hash,
                dtype=np.uint32,
            )
            for k in range(blake3.digest_size // bytes_per_lines):
                random_uint32 = random_uint32_array[k]
                first_idx = random_uint32 & self.rank_mask
                second_idx = first_idx ^ (
                    1 + np_mul_hi_u32(np.uint32(self.noise_rank - 1), random_uint32)
                )
                permutation = torch.zeros(self.noise_rank, dtype=torch.int8)
                permutation[first_idx] = 1
                permutation[second_idx] = -1
                assignment_index = i * blake3.digest_size // bytes_per_lines + k
                if assignment_index >= required_lines:
                    break
                if assign_columns:
                    noise_matrix[:, assignment_index] = permutation
                else:
                    noise_matrix[assignment_index, :] = permutation

        return noise_matrix
