# Changelog

joules is published on PyPI as [landauer-gap](https://pypi.org/project/landauer-gap/). Newest first.

## 0.8.2 (3 November 2026)

- **AMD GPUs on Windows.** joules now reads AMD GPU power on Windows through ADLX, the library that ships with every
  AMD graphics driver (`amdadlx64.dll`): total board power where the driver reports it, else GPU power (common on
  built-in GPUs such as the Ryzen AI Max+ chips' Radeon 8060S/8065S). It is sampled every 0.1 s and added up, labelled
  `adlx power sampling`, and shows in `joules doctor`. On a built-in GPU joules notes that the reading is the
  driver's GPU power, not necessarily the whole package; a wall meter gives the whole machine. Tested against a fake
  ADLX driver built from the SDK's function tables; reports from real AMD Windows machines are welcome.
- AMD GPUs without a `product_name` file, common for cloud virtual functions, are now named by their PCI IDs:
  AMD Instinct MI300/MI325/MI350 IDs are built in, other cards come from the system's `pci.ids`. A DigitalOcean
  MI300X VF used to show as "GPU 0 AMD GPU" and is now "GPU 0 AMD Instinct MI300X VF". On that card, under a 30 s
  bf16 matmul, joules read 741.5 W against 740.7 W from `rocm-smi` over the same window.

## 0.8.1 (28 October 2026)

- **Tokens per second on Live.** `joules agent --track` now reads the token counters of a vLLM or SGLang server on
  the same machine (port 8000 or 30000, or the page `--serve-metrics URL` names; `off` to not look) and sends, with
  each heartbeat, the output and prompt tokens per second over the last 30 seconds, the requests running and waiting,
  and the average time to the first token. Live turns them into speed per request and cost per million tokens at the
  machine's GPU-hour price. Only counts and times are read: never prompts, answers or model names. A WattOps server
  from before this gets power only, as before.
- A Live heartbeat still on its way when the agent stops no longer prints "Live heartbeat failed".
- Macs: a sample whose Combined line appears twice is still counted once (a guard proposed by Eric Spencer, HorneSci,
  with eight offline parser cases; real powermetrics output prints it once, so 0.8.0 readings are unaffected).
- `joules agent` on a shared GPU: a job that holds the card but whose SM time rounds to 0 (a process that keeps
  memory and runs only a tiny kernel now and then) now gets its 0 J share by SM time and says so (`attribution:
  sm_time`), where it used to read `cpu_time`, as if it had never been on the GPU. Its energy was already right.
  Seen on an H100 NVL with the shared-GPU test notebook.
- `joules` no longer asks about usage counts in a notebook's `!` cell (Kaggle, Colab, Jupyter). Those cells run in a
  pseudo-terminal that nothing typed in the notebook reaches, so the question hung the cell, for example the first
  `!joules doctor`. Use `!joules telemetry on` (or `off`) there instead.
- `joules.connect()` in a notebook run with no one to type a key (Kaggle's Save Version, or any run without an input
  box) now stops with a `ConnectError` that says to pass `key=` or set `$LANDAUER_API_KEY`, instead of failing inside
  the input box.

## 0.8.0 (24 October 2026)

- **Macs (Apple silicon): each powermetrics sample is now counted once.** With the `cpu_power,gpu_power`
  samplers, macOS prints a second "GPU Power" line after a sample's Combined (CPU + GPU + ANE) line; joules
  added that line again as a sample of its own, so five samples became nine additions and energy on Apple
  silicon read high by about the GPU's share. Found and reproduced by Eric Spencer (HorneSci). Measurements
  on Apple silicon with earlier versions should be taken again; NVIDIA, AMD, Intel and Intel-Mac readings
  are not affected.
- `joules compare --subtract-idle` measures idle again after the last run and takes each run's idle on a straight
  line between the two, so a machine that settles during a long comparison no longer has the starting idle taken
  off every run. It says so when idle moved by more than 10%. The verdict now also gives the change in average
  power and in energy above idle, with its own p-value: a change can draw more power and still use less energy.
- Callbacks for the training loops people already use: `joules.integrations.huggingface` (`attach(trainer)` or
  `JoulesCallback` for transformers' Trainer) and `joules.integrations.lightning` (`JoulesCallback`, for `lightning`
  and `pytorch_lightning`). They add `energy/joules`, `energy/kwh`, `energy/avg_watts` and `energy/joules_per_step`
  to the logs, measure each epoch as a phase, print one line when training ends, keep the numbers in `.result` and
  save the run with `track=`. Only the local main process measures. A machine with nothing to measure, a bad
  argument or a failed upload is one line on stderr, never an error in training. joules still has no dependencies.
- `joules calibrate --wall-csv log.csv` takes a logging wall meter's time-stamped readings and integrates them over
  exactly the seconds it measured; `--report run.json --wall-csv log.csv` does the same later, for a meter whose log is
  exported afterwards. A log that covers less than 95% of the run is refused. Calibrate reports keep their start and
  end times.
- `joules agent` splits a shared NVIDIA GPU by each process's SM time, from the driver's per-process utilisation
  samples or, with accounting mode on, its per-process totals. It falls back to the memory each process held only when
  the driver gives neither. A job alone on a GPU gets all of it.
- Every finished job, tracked run and report says how its energy was split (`attribution`: whole GPU, SM time, busy
  time, memory share, CPU time or CPU guess) and what share of its wall time was sampled (`coverage`). The agent warns
  below 95%. `joules run` marks its runs whole GPU (or whole device). A server from before these fields gets the run
  without them.
- `joules telemetry`: anonymous usage counts, off unless you say yes. The first time a command finishes in a
  terminal, joules asks once (Enter means no). With yes, once a day it sends a random ID, the version, the OS, the
  GPU maker and how many commands ran; never commands, paths, files, names or results. Never asked or sent in CI,
  with `DO_NOT_TRACK=1` or `JOULES_TELEMETRY=0`, or outside a terminal. `joules telemetry off` stops it.

## 0.7.11 (20 October 2026)

- `joules agent`, `joules connect` and `joules proxy` start at once on a Mac or VM whose DNS is slow. Python's
  HTTP server looked up the name of the address it listens on, which could hang for half a minute or more before
  the first reading reached Live.
- `joules compare` on Windows takes quoted paths, e.g. `"C:\Program Files\Python\python.exe" train.py`. The
  quotes used to stay on the path, so the program was not found.

## 0.7.10 (18 October 2026)

- Live is on every plan now, up to a number of GPUs at once: 2 on Free, 5 on Pro, every GPU on Max. A machine
  without a GPU counts as one. When a machine would go over the limit, `joules agent` prints the server's reason once
  ("Live on Free shows up to 2 GPUs at once; 2 GPUs are on Live already…") and tries again every minute. It joins
  once another machine has been off for a minute. Finished jobs still go to Tracking meanwhile.
- `joules connect` reports that case as "not on Live: …" with the same reason.

## 0.7.9 (18 October 2026)

- `joules connect` in a terminal, or `import joules; joules.connect()` in a notebook's Python cell, puts the
  machine on WattOps under one name. It asks for the API key in a hidden field, or a masked box in a notebook,
  unless one is already saved or in `$LANDAUER_API_KEY`. It checks and saves the key; the key never appears in
  output or on a command line. Then:
  - it runs the one-minute GPU test for the GPU leaderboard (`--no-board` / `board=False` skips it);
  - it starts `joules agent` in the background, so the machine's power and jobs show on Live (Max plan) and its
    finished jobs go to Tracking, in a project named after the machine.
- The default name is where the machine runs plus its GPUs: `kaggle-2x-t4`, `colab-t4`, `runpod-h100`,
  `aws-8x-a100`, or `8x-a100` on your own hardware. It is never the host name. `--name` / `name=` picks one.
- A name another machine is already sending to Live gets a number (`kaggle-t4-2`), and connect says so.
- Running connect again replaces the agent it started before. `joules connect --stop` or `.stop()` stops it.
- As root on a machine that runs systemd, connect installs the agent as the `joules-agent` service, which starts
  with the machine. It uses the same unit and `/etc/default/joules-agent` file as the cluster setup and keeps any
  other settings already in that file.
- In a notebook's `!` cell, which cannot take typing, `joules connect` without a saved key points to the Python
  cell instead.
- `joules login` under sudo now saves the key so the user can read it without sudo.

## 0.7.8 (17 October 2026)

- `--lock-clocks` on a machine that refuses clock changes even to root (containers and cloud notebooks: a Kaggle T4
  does) now says so, instead of suggesting sudo, which cannot help there. `joules doctor` names those hosts too.

## 0.7.7 (17 October 2026)

- `joules compare --lock-clocks MHZ` locks the SM clock of every GPU in `--gpus` (`nvidia-smi -i N -lgc MHZ,MHZ`)
  before the first run, warm-ups included; `--lock-clocks base` uses the card's base clock as the driver reports it.
  The clocks are always unlocked afterwards (`nvidia-smi -i N -rgc`), also after a failed run or Ctrl-C. Needs root.
  If a GPU refuses (permissions, MIG, an unsupported driver), joules stops before running anything and says why.
- Works with `--limits` (clocks locked first, then the power limits stepped) and `--cool-to`.
- The locked clock is in the table header, `--markdown`, `--csv` (`locked_mhz`, and each run's average
  `gpu_sm_clock_mhz`) and the label of each run sent with `--track` (`fp16 @ 1590 MHz locked`). Unlocked comparisons
  say "clocks not locked".
- With locked clocks, the throttling warning becomes whether the GPU held the lock: if it still dropped below it
  under one side (its power or thermal cap), joules names the side and its average clock.
- `joules doctor` says whether clock locking is permitted on the machine. It only asks; it never changes a clock.

## 0.7.6 (17 October 2026)

- `joules bench gpu --share`: the same one-minute test on any GPU (8192 × 8192 matrix multiplications in fp16 and
  fp32) for the GPU leaderboard at wattops.dev/leaderboard.
- `joules bench --share` measures the idle machine for 5 seconds first, so the per-token board compares energy
  above idle everywhere.

## 0.7.5 (17 October 2026)

- A `joules.Session` in a notebook is called "notebook" on Tracking; a 0 J phase is explained; a GPU count is
  shown once ("2× Tesla T4").

## 0.7.4 (17 October 2026)

- `joules.Session(track="project")` saves the whole session, with each phase, to Tracking.

## 0.7.3 (17 October 2026)

- `joules compare`'s start-temperature note says what is true and what helps (`--cool-to` only waits for a card to
  cool; add warm-up rounds).

## 0.7.2 (16 October 2026)

- `joules agent` sends one reading per GPU for the Live page.

## 0.7.1 (16 October 2026)

- Finds the NVIDIA driver library where the loader does not look.

## 0.7.0 (16 October 2026)

- `joules compare`: the same work done two or more ways, interleaved, with medians, spread and Mann-Whitney
  p-values; `--limits` and `--cool-to`. Phase-level energy with `joules.Session` and the agent's phase marks.
