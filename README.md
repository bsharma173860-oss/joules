# joules

[![tests](https://github.com/bsharma173860-oss/joules/actions/workflows/tests.yml/badge.svg)](https://github.com/bsharma173860-oss/joules/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/landauer-gap?label=pip%20install%20landauer-gap)](https://pypi.org/project/landauer-gap/)
[![license: MIT](https://img.shields.io/badge/license-MIT-555)](LICENSE)

**Like `time`, but for energy.** Wrap any command and get the real GPU, CPU and DRAM energy it
used, what that cost, its carbon, and how far it sits above the Landauer limit of physics.
Benchmark a local model and get joules per token, and how that compares with an API's price.

Illustrative output:

```
$ joules -- python train.py
joules · python train.py · 2.31 h · exit 0
  GPU 0 NVIDIA H100 80GB HBM3   4.91 MJ    590 W avg  nvml counter
  CPU package 0                  612 kJ     74 W avg  rapl counter
  DRAM (package 0)              98.7 kJ     12 W avg  rapl counter
  ──────────────────────────────────────────────────────────
  total  5.62 MJ = 1.56 kWh
  cost 0.156 at 0.1/kWh · CO₂ 577 g at 370 g/kWh
```

```
$ joules bench ollama:llama3.1:8b --api-price 0.20
  per token      3.74 J   ·   1.04 kWh per 1M tokens
  electricity    0.104 per 1M output tokens at 0.1/kWh
  vs API         0.2 per 1M → electricity is 1.9× cheaper than the API (hardware cost not included)
```

No dependencies: only the Python standard library and your GPU driver.

On NVIDIA GPUs every measurement also reads the card's clocks, temperature, power limit, busy and memory-controller
percentages and the driver's throttle reasons, and says in words what they mean:

```
  GPU 0 NVIDIA H100 80GB HBM3   4.91 MJ    590 W avg  nvml counter
    1,412 MHz (71% of max) · 94% busy · memory 38% · 83 °C max · limit 700 W · compute-bound · throttled 62% of the time (thermal)
```

Throttled means below the maximum clock for a reason other than having nothing to do: the power cap, heat, a hardware
slowdown or a power brake. Compute-bound or memory-bound is a coarse reading from the two busy percentages, and is
labelled as such. `--track` sends these as numbers, so the Tracking page shows which runs were throttled.

## Install

macOS and Linux:

```bash
curl -LsSf https://wattops.dev/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
joules doctor                                # what can be measured here, with a live reading from each meter
```

Windows 10 or 11, in PowerShell (no admin rights):

```powershell
irm https://wattops.dev/install.ps1 | iex
joules doctor
```

The installers need only Python 3.9+ (no sudo, no pip) and update joules when run again. Or use
pipx: `pipx install landauer-gap` (update: `pipx upgrade landauer-gap`).

## Commands

| Command | What it does |
|---|---|
| `joules -- CMD …` / `joules run -- CMD …` | Measure a command. Its exit code passes through, so it works in CI. |
| `joules bench ollama:MODEL` | Energy per token of a model served by Ollama. joules starts Ollama if it is installed but not running, and downloads the model the first time. |
| `joules bench openai:MODEL --url http://host:8000/v1` | Same for any OpenAI-compatible server: vLLM, llama.cpp, LM Studio, TGI. |
| `joules run --track PROJECT -- CMD …` | Measure and save the run to your private [Tracking page](https://wattops.dev/#/track). Also on `bench`. |
| `joules bench … --share` | Add the result to the public [leaderboard](https://wattops.dev/leaderboard). It measures the idle machine for 5 seconds first. |
| `joules bench gpu --share` | The same one-minute test on any GPU (8192 × 8192 matrix multiplications in fp16 and fp32, through the PyTorch already installed) for the GPU leaderboard. |
| `joules share FILE` | Share a result saved earlier with `--out FILE`. |
| `joules ci --budget 10 -- CMD …` | Energy check for a pull request: this change vs the base branch. |
| `joules compare "CMD A" "CMD B"` | The same work done two or more ways: interleaved runs after a warm-up, median energy, time, power and energy × time with their spread, a Mann-Whitney p-value against the baseline, and a warning when the GPU did not run under the same conditions. `--limits 100,70` repeats it at fixed power limits (sudo), `--lock-clocks base` (or a clock in MHz) locks the GPU's SM clock for every run and unlocks it afterwards (sudo), `--cool-to 50` starts every run from the same temperature, `--markdown`/`--csv` for papers. |
| `joules proxy --upstream URL` | Energy receipt on every response of a model server. |
| `joules agent` | Energy per job on a GPU server or cluster node: Slurm jobs, Kubernetes pods, containers, users. |
| `joules connect` | Put this machine on WattOps under one name: the GPU leaderboard, Live and Tracking. Asks for the key without showing it. In a notebook: `import joules; joules.connect()`. |
| `joules tune ollama:MODEL` | Benchmark at several GPU power limits (100, 85, 70, 55% of the maximum) and report the one with the cheapest token that is at most `--max-slowdown` (10%) slower. Puts the limit back. Needs `sudo`. |
| `joules calibrate --wall-wh X` | Compare the hardware meters with a wall-plug meter over the same time, and record what share they cover. With a logging meter, `--wall-csv log.csv` integrates its time-stamped readings over exactly the seconds measured (or later: `--report run.json --wall-csv log.csv`). |
| `joules devices` | List the meters joules found, and why any are missing. |
| `joules doctor` | Check this machine: every meter with a live reading, GPU drivers, CPU counters, API key, connection, Ollama. The report has no key, machine name, user name or home folder, so it can be shared as it is. |
| `joules telemetry [on\|off]` | Anonymous usage counts, off unless you say yes. Shows exactly what is sent. |
| `joules login KEY` / `joules logout` | Save your API key on this computer, or remove it. After login every `joules run` and `joules bench` on this computer appears on your Tracking page (project `default`) without `--track`; `--no-track` or `JOULES_NO_TRACK=1` keeps a run local. A key from `--key` or `$LANDAUER_API_KEY` (scripts, CI) tracks only with `--track`. |

Useful options: `--json` / `--out FILE` (machine-readable result), `--subtract-idle 5` (measure the
idle machine first and also report energy above idle), `--gpus 0,1`, `--price`, `--grid fr` or
`--ci 55`, `--pue 1.2`.

## Tracking your runs over time

```bash
joules login lgk_…                              # once: saves your key (Settings → API keys shows this line)
joules run --track llama-ft --params 8 --tokens 2 -- python train.py
joules --track nightly -- ./train.sh            # short form
```

Each tracked run appears on the Tracking page within seconds: Landauer gap over time, energy per run
and totals for the project. joules sends the energy, time, cost, CO₂, hardware, operations and gap,
and a short label: the program and script name (`python train.py`) or your `--label`. It never sends
arguments, inline code, outputs or file contents. A tracking problem is reported but never changes
the command's exit code, so it is safe in CI.

## Comparing two ways of doing the same work

```bash
joules compare --names new,old --runs 7 --subtract-idle 10 --markdown results.md \
  "./bench --runtime new" "./bench --runtime old"
```

Runs a warm-up round, then the commands interleaved (A B A B, or `--order abba`), and reports per command the
median energy, time, average power and energy × time with the spread of the middle half of the runs, the GPU's
clock, throttling and temperature, and against the baseline the change with a Mann-Whitney p-value. With `--subtract-idle` it measures idle before and
after the runs and gives energy above idle too, with its own p-value. It warns when
one side was throttled more, ran at a different clock or started hotter, and `--limits 100,85,70,55` (sudo)
repeats the comparison at fixed GPU power limits so a gain cannot come from the card boosting differently. The
full method is in [docs/BENCHMARKING.md](https://github.com/bsharma173860-oss/d3-research/blob/main/apps/landauer-gap/docs/BENCHMARKING.md).

## Phases of your own program

Inside a Python program:

```python
import joules

with joules.Session() as s:
    with s.phase("incremental update") as p:
        runtime.update(batch)
    print(p.energy_j, p.duration_s, p.gpu)
    print(s.summary())          # per phase name: count, median energy and time, quartiles
```

From any language, with `joules agent` running on the machine:

```bash
curl -s -X POST localhost:9877/phase/start -H 'content-type: application/json' -d '{"name": "incremental update"}'
curl -s -X POST localhost:9877/phase/end   -H 'content-type: application/json' -d '{"id": "p1"}'
curl -s localhost:9877/phases               # medians per phase name
```

Each mark reads every meter at that moment, so a phase's energy is exact to the counter's resolution. Phases can
nest and overlap, and the end of each one also says what the GPU did during it (clock, throttling, temperature at
start and end). The phase endpoints refuse requests from web browsers.

## In your training loop

Hugging Face Trainer and PyTorch Lightning get a callback each. joules still needs nothing installed; each callback
needs only its own framework.

```python
from joules.integrations.huggingface import attach

trainer = Trainer(model=model, args=args, train_dataset=ds)
energy = attach(trainer)        # put first, so the energy reaches W&B, TensorBoard and every report_to logger
trainer.train()
print(energy.result)            # energy_j, kwh, avg_w, j_per_step, cost, co2_g, per-epoch phases, GPU health
```

```python
from joules.integrations.lightning import JoulesCallback

energy = JoulesCallback()
L.Trainer(max_epochs=3, callbacks=[energy]).fit(model, data)
```

Both add `energy/joules`, `energy/kwh`, `energy/avg_watts` and `energy/joules_per_step` to the logs, so the
energy is charted next to the loss; each epoch is a phase; when training ends they print one line and keep the
numbers in `.result`. `track="my-project"` saves the run to Tracking. Only the local main process measures, so a
machine's GPUs are counted once. With nothing to measure on the machine, the callback says so in one line and
training runs as before. `JoulesCallback` also works in `Trainer(callbacks=[...])`, but the Trainer calls it after
its loggers, so there the energy reaches only the printed line and `.result`; `attach()` avoids that.

## Energy check on every pull request

`joules ci` runs a command on your checkout and on the base branch (a temporary git worktree),
alternating runs so drift hits both sides, and compares the medians. It writes a Markdown summary,
can post one pull request comment that it updates on each push, and fails when energy grows by more
than `--budget` percent. A change smaller than the run-to-run spread is reported as noise, never as a
failure.

```bash
joules ci --base main --runs 3 --budget 10 -- pytest -q
```

In GitHub Actions use the [PR energy check action](https://github.com/bsharma173860-oss/joules-action). On
hosted runners, which have no energy counters, energy is estimated from CPU time and the comment says
so; on a self-hosted runner with a GPU or RAPL it is measured. `joules run --estimate` uses the same
estimate when a machine has no counters.

## Energy receipts for AI responses

> **Frozen:** `joules proxy` gets bug fixes only, no new features, until a paying user asks for more.

`joules proxy` sits in front of a model server and measures the hardware while each request runs.
Point your client at the proxy instead of the server; nothing else changes.

```bash
joules proxy --upstream http://localhost:11434          # Ollama
joules proxy --upstream http://localhost:8000 --subtract-idle 5   # vLLM, measured above idle
```

```
$ curl -si localhost:8787/v1/chat/completions -d '{"model":"llama3.1:8b","messages":[…]}'
X-Energy-Joules: 41.2
X-Energy-Output-Tokens: 212
X-Energy-Joules-Per-Token: 0.194
X-Energy-CO2-Grams: 0.00423
X-Energy-Receipt: /joules/receipts/r-000017
```

- Streaming responses (SSE or Ollama's NDJSON) pass through untouched and carry `X-Energy-Receipt`;
  the receipt at that address is complete when the stream ends.
- Tokens come from the response (`usage`, or Ollama's `eval_count`). Without them, streamed pieces
  are counted and the receipt says `tokens_exact: false`.
- Meters measure whole devices, so in each sampling interval the energy is shared equally between
  the requests running at that moment. With `--subtract-idle` receipts also carry energy above idle.
- `/joules/receipts` lists the newest receipts, `/joules/metrics` serves Prometheus metrics, and
  `/joules/health` says what is measured.
- `--track PROJECT` sends one summary per model every 5 minutes to your Tracking page: never
  prompts or outputs.
- It listens on 127.0.0.1 by default. `--host 0.0.0.0` exposes it, and your model server, to your
  network.

## Connect a machine in one step

```bash
joules connect                    # asks for your API key (hidden), then does everything below
joules connect --name lab-a100-1  # pick the name; --stop stops it
```

In a Kaggle, Colab or Jupyter notebook, use a Python cell, which shows a masked box for the key:

```python
!pip install -q -U landauer-gap     # first cell
import joules; joules.connect()     # second cell: paste the key into the box (never into the notebook)
```

connect checks the key and saves it like `joules login`. It names the machine after where it runs and its
GPUs (`kaggle-2x-t4`, `colab-t4`, `runpod-h100`, `aws-8x-a100`, or `8x-a100` on your own hardware), never
its host name. It then:

1. runs the one-minute GPU test and puts it on the GPU leaderboard (skip it with `--no-board` / `board=False`);
2. starts `joules agent` in the background under that name. Each program that runs a minute or more and then
   ends goes to your Tracking page, in a project named after the machine. The machine's power and jobs also
   show on Live every 5 seconds: 1 GPU on the Free plan, more paid per GPU (from $4 a GPU a month).

As root on a machine that runs systemd, connect installs the agent as the `joules-agent` service instead, so it
starts with the machine (`sudo joules connect`). Elsewhere (containers, notebooks, without root) the agent runs
until the machine or session stops; run connect again after a restart.

Code that runs in a notebook's own Python counts as one job until the session ends. To track a cell, wrap it in
`with joules.measure("train", track="kaggle-2x-t4"):`.

If another machine is already sending to Live under that name, connect adds a number (`kaggle-t4-2`) and says so.
On Kaggle, keep the key in Add-ons → Secrets and pass it: `joules.connect(key=UserSecretsClient().get_secret("WATTOPS_KEY"))`.

## Energy per job on GPU clusters

`joules agent` runs on every node of a cluster, measures it every second and splits each device's
energy between the jobs on it. A job alone on a GPU gets all of it. A shared NVIDIA GPU is split by each
process's SM time (from the driver's per-process utilisation, or accounting mode), and by the memory each
process holds only when the driver gives neither. AMD and Intel GPUs are split by GPU time, CPUs and DRAM by
CPU time. Every finished job says which split it got and what share of its time was sampled (warned below
95%); [docs/BENCHMARKING.md](../docs/BENCHMARKING.md#per-job-energy-on-a-shared-machine-when-the-split-is-exact)
says when the split is exact. Jobs are Slurm jobs, Kubernetes pods, containers or users' programs. GPUs a
job holds but barely uses are reported as idle GPU energy.

```bash
sudo joules agent                         # http://127.0.0.1:9877/jobs and /metrics (Prometheus)
sudo joules agent --track cluster         # each finished job on your Tracking page
```

`--track` also puts the node on the **Live** page (https://wattops.dev/#/live): 1 GPU on the Free plan, more
paid per GPU (from $4 a GPU a month), every GPU on Fleet (a machine without a GPU counts as one; a machine over the limit waits, trying every minute). Live shows
its power, cost per hour, idle GPUs and the jobs running now, updated every 5 seconds (`--live-every`,
0 turns it off), and every GPU on its own: power, how busy, clock, temperature, power limit and whether it is
throttled and why, so a fleet of a hundred GPUs shows which ones are idle, hot or held back. Heartbeats do not count against your API calls. On a Mac the agent measures the whole
machine (`joules agent --track fleet`), so you can watch it live too. A sleeping machine sends nothing,
so Live shows a gap until it wakes; `--keep-awake` keeps it from sleeping while the agent runs (closing a
laptop's lid still puts it to sleep).

A machine serving a model with **vLLM or SGLang** also shows what it serves: tokens per second, the requests running
and waiting, the speed each request gets, the time to the first token, and what a million tokens cost at the
machine's GPU-hour price, beside an API price you set to compare with. The agent reads the server's own counters at
port 8000 (vLLM) or 30000 (SGLang); `--serve-metrics URL` names another address, `--serve-metrics off` stops it. Only
counts and times are read, never prompts, answers or model names.

A Kubernetes DaemonSet, a systemd unit for Slurm nodes, Prometheus scrape and alert rules and a
Grafana dashboard are available: [ask us](https://wattops.dev/#contact=max).

## The leaderboards

[wattops.dev/leaderboard](https://wattops.dev/leaderboard) is public: anyone can look, no account needed.

**GPUs.** One fixed test on every card, so cards can be compared:

```bash
joules login lgk_…              # once; free key: wattops.dev → Settings → API keys
joules bench gpu --share        # about a minute; in a Kaggle or Colab notebook: !joules bench gpu --share
```

It multiplies 8192 × 8192 matrices for 20 seconds in fp16 and 20 in fp32 (full fp32: TF32 is switched
off) after a warm-up, with the energy from the GPU's own meter over exactly that stretch, and reports
TFLOP/s, watts and GFLOP per joule, with the card's clock, throttling and temperature. It uses the
PyTorch already installed: in the same Python as joules, or else `python3` on the PATH (a conda env
or venv), with joules brought along. `--gpus N` picks the GPU (PyTorch's numbering). joules checks
that the meter it reads is the card PyTorch used, and that the card was busy. It sends the GPU's name,
its power limit, which meter was used, and per precision the number of multiplications, the energy,
the time, the share of time throttled, the clock and the end temperature. The server works out the
rest, and refuses results no card could produce.

**Local models.**

```bash
joules bench ollama:llama3.1:8b --share
```

Machines are ranked against each other on the same model (never one model against another), by
energy per output token above idle: `--share` measures the idle machine for 5 seconds first. Each
person's runs give one median, so one person counts once; a row is the median of those, and rows
whose meters cover different things (GPU only, GPU + CPU, a Mac's whole chip) are kept apart.
joules sends only the model name, hardware name and count, energy per token, tokens per second,
average power, output tokens, whether idle power was subtracted, which meters were used and the
joules version.

Neither sends prompts, outputs, host names or paths. Simulated runs are refused, and you can remove
your own results on the Leaderboard page.

## Estimate vs measured, and calibration

Tell joules the size of the training job and it compares the measurement with the WattOps
estimate, then works out what your hardware really achieved:

```
$ joules run --params 1 --tokens 2 -- python finetune.py        # 1B model, 2B tokens, one H100
  estimate vs measured  17 MJ estimated (40% MFU, 80% draw) → 14.8 MJ measured (-13%)
  your hardware ran at 42.5% effective utilisation and 74% of rated power
  open in the calculator with these numbers: https://wattops.dev/?s=…
```

The link opens the website calculator with your measured utilisation and power draw, so every
future estimate starts from your real hardware instead of a textbook default. If the job you
describe could not fit in the measured time, joules says so instead of reporting nonsense.

## What it measures, and how

| Hardware | Source | Kind |
|---|---|---|
| NVIDIA GPUs | NVML (driver library, via ctypes) | energy counter on Volta and newer; power sampling on older cards |
| AMD GPUs (Instinct, Radeon) | amdgpu hwmon (`energy1_input`, else `power1_average`) | counter or sampling |
| AMD GPUs on Windows (Radeon, Ryzen AI built-in) | ADLX, the AMD driver's `amdadlx64.dll` (total board power, else GPU power) | sampling |
| Intel GPUs (Arc, Flex, Max) | i915 / xe hwmon `energy1_input` | energy counter |
| Intel and AMD CPUs, DRAM | RAPL (`/sys/class/powercap`) | counter, wraparound handled |
| Apple Silicon Macs | `powermetrics` (asks for your password once; the command still runs as you) | sampling: CPU + GPU + Neural Engine |
| Intel Macs | `powermetrics` (asks for your password once) | sampling: CPU package (cores, integrated GPU, DRAM) |
| Windows | NVML for NVIDIA GPUs, ADLX for AMD GPUs; the CPU is estimated from CPU time (Windows gives programs no CPU energy counter without a driver) | GPU counter; CPU estimate, labelled |

Where there is no counter (cloud VMs, CI runners, Windows CPUs), joules estimates energy from CPU time and
labels every such result ESTIMATE; estimates are never added to the leaderboard.

Meters cover whole devices, so anything else running on the same GPU or CPU is included. Use
`--subtract-idle`, and keep the machine otherwise quiet, for the cleanest numbers. Power supply
losses, fans, networking and cooling are not measured; `--pue` adds a facility overhead.

joules finds the NVIDIA driver's library where the loader looks and also in the folders drivers are known to use (Google Colab and Kaggle, Debian and Ubuntu, RHEL, NVIDIA's container toolkit, WSL 2, NixOS); if it still cannot, `joules doctor` says so and `JOULES_NVML_LIB=/path/to/libnvidia-ml.so.1` names it.

`CUDA_VISIBLE_DEVICES` is respected, with indices, GPU UUIDs (as Kubernetes and Slurm set them) or MIG
slices (MIG measures the whole GPU). On AMD, `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` and `--gpus` pick cards. Reading RAPL on recent Linux kernels needs root, or
`sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj`.

## Usage counts

The first time a command finishes in a terminal, joules asks once: *Send anonymous usage counts to help improve joules? [y/N]*. Enter means no, and nothing is sent. With yes, once a day it sends a random ID made on your computer, the joules version, the OS, the GPU maker and how many commands ran. Never commands, arguments, paths, files, host or user names, results or your key. It never asks or sends in CI, with `DO_NOT_TRACK=1` or `JOULES_TELEMETRY=0`, or outside a terminal (pipes, notebooks, services). `joules telemetry` shows the exact record; `joules telemetry off` stops it.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The real sysfs layouts are rebuilt in a temporary directory (including counter wraparound), the
NVML binding runs against a compiled stand-in for `libnvidia-ml.so.1`, and model servers are
local HTTP stand-ins. `JOULES_SIMULATE="gpu:H100:650,cpu:package 0:80"` runs everything on
constant simulated power for demos; results are always labelled SIMULATED.
