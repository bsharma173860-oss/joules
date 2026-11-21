/* Stand-in for libnvidia-ml.so.1 on a laptop: one GPU (a GeForce RTX 3050 Laptop GPU) with NO energy counter, only
   power readings, as on many laptop and consumer cards. The power in milliwatts is read from the file named by
   FAKE_NVML_POWER_FILE at every call, so a test script can change it phase by phase. FAKE_NVML_FAIL_EVERY=N makes
   every Nth power read fail, as a driver sometimes does. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef void *nvmlDevice_t;
static unsigned long calls = 0;
int nvmlInit_v2(void) { return 0; }
int nvmlShutdown(void) { return 0; }
int nvmlDeviceGetCount_v2(unsigned int *n) { *n = 1; return 0; }
int nvmlDeviceGetHandleByIndex_v2(unsigned int i, nvmlDevice_t *h) { if (i > 0) return 2; *h = (nvmlDevice_t)1L; return 0; }
int nvmlDeviceGetName(nvmlDevice_t h, char *buf, unsigned int len) { (void)h; strncpy(buf, "NVIDIA GeForce RTX 3050 Laptop GPU", len); return 0; }
int nvmlDeviceGetTotalEnergyConsumption(nvmlDevice_t h, unsigned long long *mj) { (void)h; (void)mj; return 3; }  /* NOT_SUPPORTED */
int nvmlDeviceGetPowerUsage(nvmlDevice_t h, unsigned int *mw) {
  (void)h; const char *f = getenv("FAKE_NVML_POWER_FILE"), *e = getenv("FAKE_NVML_FAIL_EVERY");
  calls++;
  if (e && atoi(e) > 0 && calls % (unsigned long)atoi(e) == 0) return 999;   /* NVML_ERROR_UNKNOWN */
  FILE *fp = f ? fopen(f, "r") : NULL; unsigned int v = 0;
  if (!fp) return 999;
  if (fscanf(fp, "%u", &v) != 1) { fclose(fp); return 999; }
  fclose(fp); *mw = v; return 0;
}
typedef struct { unsigned int gpu, memory; } util_t;
int nvmlDeviceGetUtilizationRates(nvmlDevice_t h, util_t *u) { (void)h; u->gpu = 50; u->memory = 20; return 0; }
int nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t h, unsigned int *n, void *p) { (void)h; (void)p; *n = 0; return 0; }
int nvmlDeviceGetClockInfo(nvmlDevice_t h, unsigned int type, unsigned int *mhz) { (void)h; *mhz = type == 2 ? 6001 : 1500; return 0; }
int nvmlDeviceGetMaxClockInfo(nvmlDevice_t h, unsigned int type, unsigned int *mhz) { (void)h; *mhz = type == 2 ? 6001 : 1740; return 0; }
int nvmlDeviceGetTemperature(nvmlDevice_t h, unsigned int sensor, unsigned int *c) { (void)h; (void)sensor; *c = 61; return 0; }
int nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t h, unsigned int *mw) { (void)h; *mw = 60000; return 0; }
int nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice_t h, unsigned int *lo, unsigned int *hi) { (void)h; *lo = 35000; *hi = 60000; return 0; }
int nvmlDeviceSetPowerManagementLimit(nvmlDevice_t h, unsigned int mw) { (void)h; (void)mw; return 4; }
int nvmlDeviceGetCurrentClocksEventReasons(nvmlDevice_t h, unsigned long long *r) { (void)h; *r = 0; return 0; }
typedef struct { unsigned long long total, free, used; } mem_t;
int nvmlDeviceGetMemoryInfo(nvmlDevice_t h, mem_t *m) { (void)h; m->total = 4ULL << 30; m->used = 3ULL << 30; m->free = m->total - m->used; return 0; }
int nvmlSystemGetDriverVersion(char *buf, unsigned int len) { strncpy(buf, "560.35.03", len); return 0; }
