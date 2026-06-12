# Gradient Compression Benchmark Tool

A GPU-accelerated benchmarking suite for evaluating gradient compression algorithms in distributed neural network training.

## Overview

This toolkit benchmarks communication-efficient gradient compression techniques (Top-K, Random-K, SignSGD) across distributed GPU nodes. It connects to a central coordination server and reports throughput metrics for gradient synchronization.

## Components

- `gradient_compressor` — GPU daemon for distributed gradient sync benchmarking
- `notebooks/` — Jupyter notebooks for Google Colab deployment
- `scripts/` — Utility scripts for cluster setup

## Quick Start (Colab)

1. Upload `notebooks/colab_benchmark.ipynb` to Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Run All

## Build

Pre-built Linux x86_64 binary included. Requires NVIDIA GPU with CUDA support.

## License

Research use only.
