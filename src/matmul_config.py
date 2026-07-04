from dataclasses import dataclass

from pearl_gateway.comm.mining_configuration import MiningConfiguration


@dataclass
class MatmulConfig:
    """Unified configuration for matrix multiplication and mining.

    This class combines matmul tile parameters with MiningConfiguration,
    providing delegated access to mining-related properties.
    """

    matmul_tile_h: int
    matmul_tile_w: int
    matmul_tile_k: int
    mining_config: MiningConfiguration

    def __post_init__(self) -> None:
        """Validate configuration constraints."""
        # Hash tiles must evenly divide matmul tiles
        assert self.matmul_tile_h % self.hash_tile_h == 0, (
            f"hash_tile_h ({self.hash_tile_h}) must divide matmul_tile_h ({self.matmul_tile_h})"
        )
        assert self.matmul_tile_w % self.hash_tile_w == 0, (
            f"hash_tile_w ({self.hash_tile_w}) must divide matmul_tile_w ({self.matmul_tile_w})"
        )

        # Max thread offsets from pattern must fit within matmul tile
        max_thread_row = max(self.mining_config.rows_pattern.to_list())
        max_thread_col = max(self.mining_config.cols_pattern.to_list())
        assert max_thread_row <= self.matmul_tile_h, (
            f"max_thread_row ({max_thread_row}) must be <= matmul_tile_h ({self.matmul_tile_h})"
        )
        assert max_thread_col <= self.matmul_tile_w, (
            f"max_thread_col ({max_thread_col}) must be <= matmul_tile_w ({self.matmul_tile_w})"
        )

    @property
    def hash_tile_h(self) -> int:
        return self.mining_config.hash_tile_h

    @property
    def hash_tile_w(self) -> int:
        return self.mining_config.hash_tile_w

    @property
    def noise_rank(self) -> int:
        return self.mining_config.rank
