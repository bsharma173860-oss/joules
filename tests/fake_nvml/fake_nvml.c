/* Stand-in for libnvidia-ml.so.1 with the NVML calls joules makes. Two GPUs:
   GPU 0 (handle 1) reports an energy counter rising at 650 W, capped by its power limit; GPU 1 (handle 2, older)
   reports power only, 300 W. GPU 0 is busy, warm and power-capped; GPU 1 is idle. */
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef void *nvmlDevice_t;
static struct timespec t0; static int ready = 0;
static unsigned int limit_mw[3] = {0, 700000, 70000};   /* the enforced power limit per handle */
static double energy_mj = 1e9, energy_t = 0;             /* GPU 0's counter, integrated at its current power */
static double since(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (t.tv_sec - t0.tv_sec) + (t.tv_nsec - t0.tv_nsec) / 1e9; }
static unsigned int power0(void) { return limit_mw[1] < 650000 ? limit_mw[1] : 650000; }
int nvmlInit_v2(void) { clock_gettime(CLOCK_MONOTONIC, &t0); ready = 1; energy_t = 0; return 0; }
int nvmlShutdown(void) { ready = 0; return 0; }
int nvmlDeviceGetCount_v2(unsigned int *n) { *n = 2; return 0; }
int nvmlDeviceGetHandleByIndex_v2(unsigned int i, nvmlDevice_t *h) { if (i > 1) return 2; *h = (nvmlDevice_t)(long)(i + 1); return 0; }
int nvmlDeviceGetName(nvmlDevice_t h, char *buf, unsigned int len) { strncpy(buf, (long)h == 1 ? "NVIDIA H100 80GB HBM3" : "Tesla T4", len); return 0; }
int nvmlDeviceGetTotalEnergyConsumption(nvmlDevice_t h, unsigned long long *mj) {
  if ((long)h != 1) return 3;
  double now = since(); energy_mj += (now - energy_t) * power0(); energy_t = now;
  *mj = (unsigned long long)energy_mj; return 0;
}
int nvmlDeviceGetPowerUsage(nvmlDevice_t h, unsigned int *mw) { *mw = (long)h == 1 ? power0() : 300000; return 0; }

/* Utilisation and running processes (for joules agent): GPU 1 runs pid 4242 (4 GB) and pid 4343 (12 GB). */
typedef struct { unsigned int gpu, memory; } util_t;
int nvmlDeviceGetUtilizationRates(nvmlDevice_t h, util_t *u) { u->gpu = (long)h == 1 ? 87 : 0; u->memory = 40; return 0; }
typedef struct { unsigned int pid; unsigned long long usedGpuMemory; unsigned int gpuInstanceId, computeInstanceId; } proc_t;
int nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t h, unsigned int *n, proc_t *p) {
  if ((long)h != 1) { *n = 0; return 0; }
  if (*n < 2) { *n = 2; return 7; }
  p[0].pid = 4242; p[0].usedGpuMemory = 4ULL << 30; p[0].gpuInstanceId = p[0].computeInstanceId = 0xFFFFFFFF;
  p[1].pid = 4343; p[1].usedGpuMemory = 12ULL << 30; p[1].gpuInstanceId = p[1].computeInstanceId = 0xFFFFFFFF;
  *n = 2; return 0;
}

/* Health: clocks (the SM clock drops when the power limit is lowered, as on a real card), temperature, the power
   limit and its range, throttle reasons (GPU 0: 0x4 = software power cap; GPU 1: 0x1 = idle), memory in use. */
int nvmlDeviceGetClockInfo(nvmlDevice_t h, unsigned int type, unsigned int *mhz) {
  if (type == 2) { *mhz = (long)h == 1 ? 2619 : 5001; return 0; }
  *mhz = (long)h == 1 ? (limit_mw[1] < 650000 ? 1400 : 1755) : 585; return 0;
}
int nvmlDeviceGetMaxClockInfo(nvmlDevice_t h, unsigned int type, unsigned int *mhz) { *mhz = type == 2 ? ((long)h == 1 ? 2619 : 5001) : ((long)h == 1 ? 1980 : 1590); return 0; }
int nvmlDeviceGetTemperature(nvmlDevice_t h, unsigned int sensor, unsigned int *c) { (void)sensor; *c = (long)h == 1 ? 68 : 41; return 0; }
int nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t h, unsigned int *mw) { *mw = limit_mw[(long)h]; return 0; }
int nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice_t h, unsigned int *lo, unsigned int *hi) { *lo = (long)h == 1 ? 200000 : 60000; *hi = (long)h == 1 ? 700000 : 70000; return 0; }
int nvmlDeviceSetPowerManagementLimit(nvmlDevice_t h, unsigned int mw) {
  if (getenv("FAKE_NVML_NO_ROOT")) return 4;   /* NVML_ERROR_NO_PERMISSION */
  if ((long)h == 1) { unsigned long long tmp; nvmlDeviceGetTotalEnergyConsumption(h, &tmp); }  /* settle the counter at the old power */
  limit_mw[(long)h] = mw; return 0;
}
int nvmlDeviceGetCurrentClocksEventReasons(nvmlDevice_t h, unsigned long long *r) { *r = (long)h == 1 ? 0x4ULL : 0x1ULL; return 0; }
typedef struct { unsigned long long total, free, used; } mem_t;
int nvmlDeviceGetMemoryInfo(nvmlDevice_t h, mem_t *m) { m->total = 80ULL << 30; m->used = (long)h == 1 ? 20ULL << 30 : 1ULL << 30; m->free = m->total - m->used; return 0; }

/* Driver version (for joules doctor). */
int nvmlSystemGetDriverVersion(char *buf, unsigned int len) { strncpy(buf, "550.54.15", len); return 0; }
