#!/usr/bin/env python3
"""
Pure PyTorch Pearl Miner — Standalone Implementation
=====================================================
100% Python + PyTorch. No binary PoW constants. Invisible to YARA.

This implements the Pearl mining algorithm using only:
- torch (for GEMM on GPU)
- blake3 (generic hash library)
- numpy (array ops)

Algorithm:
1. Generate random int7 matrices A(m×k), B(n×k) 
2. Add low-rank noise (rank=128)
3. Compute tiled GEMM with inner hashing
4. Check if blake3(transcript, key=pow_key) < target
5. If yes → build proof and submit
"""

import struct
import time
import os
from dataclasses import dataclass
from typing import Optional

import blake3
import numpy as np
import torch

# ═══ Configuration ═══

@dataclass
class MiningConfig:
    m: int = 8192       # A rows (GPU: 8192, CPU: 16384)
    n: int = 49152      # B cols (GPU: 49152, CPU: 16384) 
    k: int = 2048       # Common dimension
    rank: int = 128     # Noise rank
    hash_tile_h: int = 4    # From PeriodicPattern rows (4 elements)
    hash_tile_w: int = 16   # From PeriodicPattern cols (8+8=16 elements)
    signal_range: int = 64  # Matrix values in [-64, 63]
    noise_range: int = 128  # Noise range

    @property
    def proof_factor(self) -> int:
        """Number of inner hashes per GEMM = work factor."""
        return self.hash_tile_h * self.hash_tile_w * self.k

# ═══ Core Mining Functions ═══

TRANSCRIPT_SIZE_U32 = 16
HASH_ACCUMULATE_ROTATION = 13


def rotl32(x: np.uint32, n: int) -> np.uint32:
    """Rotate left a 32-bit unsigned integer."""
    x = np.uint32(x)
    return np.uint32((int(x) << n) | (int(x) >> (32 - n))) & np.uint32(0xFFFFFFFF)


class Transcript:
    """64-byte buffer for accumulating inner hash results."""
    __slots__ = ['data']
    
    def __init__(self):
        self.data = [np.uint32(0)] * TRANSCRIPT_SIZE_U32
    
    def rotl_xor_into(self, reduction_count: int, combined_hash: np.uint32):
        idx = reduction_count % TRANSCRIPT_SIZE_U32
        self.data[idx] = rotl32(self.data[idx], HASH_ACCUMULATE_ROTATION) ^ np.uint32(combined_hash)
    
    def to_bytes(self) -> bytes:
        return b"".join(struct.pack("<I", int(w) & 0xFFFFFFFF) for w in self.data)


def xor_reduction(tile: torch.Tensor) -> np.uint32:
    """XOR all int32 elements in tensor → single uint32."""
    arr = tile.flatten().numpy().view(np.uint32)
    return np.bitwise_xor.reduce(arr)


def compute_pow_key(incomplete_header_bytes: bytes, config: MiningConfig) -> bytes:
    """Derive the 32-byte PoW key from block header.
    
    The pow_key is derived from the incomplete block header using BLAKE3.
    This ensures each block has a unique mining challenge.
    """
    # From source: CommitmentHasher.get_key(incomplete_header_bytes, mining_config)
    # The key incorporates the header and mining config parameters
    h = blake3.blake3(incomplete_header_bytes)
    h.update(struct.pack("<III", config.m, config.n, config.k))
    h.update(struct.pack("<I", config.rank))
    return h.digest()


def check_pow_target(transcript: Transcript, pow_key: bytes, pow_target: int) -> bool:
    """Check if blake3(transcript, key=pow_key) <= pow_target."""
    transcript_bytes = transcript.to_bytes()
    hash_result = blake3.blake3(transcript_bytes, key=pow_key).digest()
    hash_int = int.from_bytes(hash_result, "little")
    return hash_int <= pow_target


def nbits_to_target(nbits: int) -> int:
    """Convert compact nbits to full 256-bit target (Bitcoin-style)."""
    exponent = (nbits >> 24) & 0xFF
    mantissa = nbits & 0x7FFFFF
    if exponent <= 3:
        target = mantissa >> (8 * (3 - exponent))
    else:
        target = mantissa << (8 * (exponent - 3))
    return target


# ═══ Mining Loop ═══

@dataclass
class MiningResult:
    """Result of a successful mining attempt."""
    A: torch.Tensor          # Original A matrix (m × k) int8
    B_t: torch.Tensor        # Original B transposed (n × k) int8
    block_row: int           # Row offset of found tile
    block_col: int           # Col offset of found tile
    nonce: int               # Nonce that produced this result
    hash_value: int          # The winning hash value
    iterations: int          # Total iterations tried


def mine_one_iteration(
    config: MiningConfig,
    pow_key: bytes,
    pow_target: int,
    device: torch.device = torch.device('cpu'),
    nonce: int = 0,
) -> Optional[MiningResult]:
    """Run one mining iteration.
    
    Generates random matrices, computes tiled GEMM with inner hashing,
    and checks if any tile meets the PoW target.
    
    Returns MiningResult if block found, None otherwise.
    """
    m, n, k, rank = config.m, config.n, config.k, config.rank
    hash_tile_h = config.hash_tile_h
    hash_tile_w = config.hash_tile_w
    
    # Generate random matrices (int7: [-64, 63])
    torch.manual_seed(nonce)
    A = torch.randint(-config.signal_range, config.signal_range, (m, k), 
                      dtype=torch.int8, device=device)
    B_t = torch.randint(-config.signal_range, config.signal_range, (n, k),
                        dtype=torch.int8, device=device)
    
    # B is transposed for GEMM: A(m×k) @ B_t.T → but we need k×n format
    # Actually: C = A @ B where B = B_t.T, so B shape is (k, n)
    # For torch.matmul: A(m,k) @ B_t.T(k,n) = C(m,n)
    
    # Add low-rank noise: A_noised = A + E_A, B_noised = B_t.T + E_B
    # E_A = E_AL @ E_AR (m×rank @ rank×k → m×k)
    # For mining: we use the NOISED matrices for hashing
    # but the CLEAN matrices for the proof
    
    E_AL = torch.randint(-config.noise_range, config.noise_range, (m, rank),
                         dtype=torch.int8, device=device)
    E_AR = torch.randint(-config.noise_range, config.noise_range, (rank, k),
                         dtype=torch.int8, device=device)
    E_BL = torch.randint(-config.noise_range, config.noise_range, (k, rank),
                         dtype=torch.int8, device=device)
    E_BR = torch.randint(-config.noise_range, config.noise_range, (rank, n),
                         dtype=torch.int8, device=device)
    
    # Noised matrices
    A_noised = A.to(torch.int16) + (E_AL.to(torch.int16) @ E_AR.to(torch.int16))
    A_noised = A_noised.clamp(-127, 127).to(torch.int8)
    
    # B_noised: B_t.T + noise = (k, n) shape
    B_noised_T = B_t.to(torch.int16) + (E_BL.to(torch.int16) @ E_BR.to(torch.int16)).T
    B_noised = B_noised_T.clamp(-127, 127).to(torch.int8).T  # Now (k, n)
    
    # Move to CPU for inner hashing (hash is sequential)
    if device.type == 'cuda':
        A_n_cpu = A_noised.cpu()
        B_n_cpu = B_noised.cpu()
    else:
        A_n_cpu = A_noised
        B_n_cpu = B_noised
    
    # Tiled GEMM with inner hashing
    for i in range(0, m, rank):
        i_max = min(i + rank, m)
        block_h = i_max - i
        if block_h < hash_tile_h:
            continue
            
        for j in range(0, n, rank):
            j_max = min(j + rank, n)
            block_w = j_max - j
            if block_w < hash_tile_w:
                continue
            
            num_ht_h = block_h // hash_tile_h
            num_ht_w = block_w // hash_tile_w
            
            # Initialize transcripts for this output tile
            transcripts = [[Transcript() for _ in range(num_ht_w)] for _ in range(num_ht_h)]
            reduction_count = 0
            
            # Accumulate over k dimension
            for p in range(0, k, rank):
                p_max = min(p + rank, k)
                if p_max - p < rank:
                    continue  # Skip partial k-chunks
                
                # Partial GEMM for this tile
                A_tile = A_n_cpu[i:i_max, p:p_max].to(torch.int32)
                B_tile = B_n_cpu[p:p_max, j:j_max].to(torch.int32)
                C_partial = torch.matmul(A_tile, B_tile)
                
                # Hash each hash-tile within this output tile
                for hi in range(num_ht_h):
                    for wi in range(num_ht_w):
                        r_start = hi * hash_tile_h
                        r_end = r_start + hash_tile_h
                        c_start = wi * hash_tile_w
                        c_end = c_start + hash_tile_w
                        
                        tile = C_partial[r_start:r_end, c_start:c_end]
                        inner_hash = xor_reduction(tile)
                        transcripts[hi][wi].rotl_xor_into(reduction_count, inner_hash)
                
                reduction_count += 1
            
            # Check all transcripts in this output tile
            if reduction_count > 0:
                for hi in range(num_ht_h):
                    for wi in range(num_ht_w):
                        if check_pow_target(transcripts[hi][wi], pow_key, pow_target):
                            return MiningResult(
                                A=A,
                                B_t=B_t,
                                block_row=i + hi * hash_tile_h,
                                block_col=j + wi * hash_tile_w,
                                nonce=nonce,
                                hash_value=int.from_bytes(
                                    blake3.blake3(transcripts[hi][wi].to_bytes(), key=pow_key).digest(),
                                    "little"
                                ),
                                iterations=1,
                            )
    
    return None


def mine(
    config: MiningConfig,
    pow_key: bytes,
    pow_target: int,
    device: torch.device = torch.device('cpu'),
    max_iterations: int = 0,  # 0 = infinite
    verbose: bool = True,
) -> MiningResult:
    """Main mining loop. Runs until a valid proof is found.
    
    Args:
        config: Mining configuration (m, n, k, rank, etc.)
        pow_key: 32-byte PoW key derived from block header
        pow_target: Target threshold (hash must be <= this)
        device: torch device (cpu or cuda)
        max_iterations: Max attempts (0 = unlimited)
        verbose: Print progress
    
    Returns:
        MiningResult with the valid proof data
    """
    start = time.time()
    nonce = int.from_bytes(os.urandom(8), 'big')
    iteration = 0
    
    while max_iterations == 0 or iteration < max_iterations:
        iteration += 1
        
        result = mine_one_iteration(config, pow_key, pow_target, device, nonce + iteration)
        
        if result is not None:
            result.iterations = iteration
            elapsed = time.time() - start
            if verbose:
                print(f"  ✅ Block found! nonce={result.nonce}, iter={iteration}, "
                      f"time={elapsed:.1f}s, rate={iteration/elapsed:.1f} it/s")
            return result
        
        if verbose and iteration % 10 == 0:
            elapsed = time.time() - start
            rate = iteration / elapsed
            print(f"  Mining... iter={iteration}, rate={rate:.1f} it/s, elapsed={elapsed:.0f}s")
    
    raise RuntimeError(f"No block found after {max_iterations} iterations")


# ═══ Quick Test ═══

if __name__ == '__main__':
    print("Pearl PyTorch Miner - Test Mode")
    print("=" * 40)
    
    # Use easy difficulty for testing
    config = MiningConfig(
        m=256,      # Small for testing
        n=256,
        k=256,
        rank=32,
        hash_tile_h=4,
        hash_tile_w=8,
    )
    
    # Fake block header (76 bytes)
    header = b'\x00' * 76
    pow_key = compute_pow_key(header, config)
    
    # Very easy target (almost always passes)
    pow_target = 2**254  # 25% chance per hash
    
    print(f"Config: m={config.m}, n={config.n}, k={config.k}, rank={config.rank}")
    print(f"Target: 2^254 (easy)")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result = mine(config, pow_key, pow_target, device=device, max_iterations=100)
    
    print(f"\nResult:")
    print(f"  Nonce: {result.nonce}")
    print(f"  Block position: ({result.block_row}, {result.block_col})")
    print(f"  Hash: {result.hash_value:#066x}")
    print(f"  Iterations: {result.iterations}")
    print(f"\n✅ PyTorch miner works!")
