"""AMD GPUs on Windows (ADLX). No AMD GPU or Windows needed: a fake ADLX driver is built from real C function
tables (ctypes callbacks laid out in the slot order of AMD's ADLX SDK headers), so joules' vtable calls run for real.
Run: PYTHONPATH=src python3 -m unittest tests.test_adlx -v"""
import ctypes
import unittest
from unittest import mock

from joules import meters

P = ctypes.c_void_p
PP = ctypes.POINTER(ctypes.c_void_p)


class FakeAdlx:
    """ADLXInitialize / ADLXTerminate plus the objects behind them: system, GPU list, GPUs, monitoring, metrics."""

    def __init__(self, gpus, init_result=0):
        self.gpus_spec = gpus          # [(name, integrated, board_w or None, gpu_w or None)]
        self.init_result = init_result
        self.keep, self.objs, self.live, self.terminated = [], {}, set(), 0
        self.system = self._obj("system", {1: (ctypes.c_int, [P, PP], self._get_gpus), 9: (ctypes.c_int, [P, PP], self._get_perf)}, refcounted=False)
        self.gpu_objs = [self._obj("gpu", {5: (ctypes.c_int, [P, ctypes.POINTER(ctypes.c_int)], self._type),
                                           7: (ctypes.c_int, [P, ctypes.POINTER(ctypes.c_char_p)], self._name)}, data=i)
                         for i in range(len(gpus))]
        init_t = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_uint64, PP)
        term_t = ctypes.CFUNCTYPE(ctypes.c_int)
        self.ADLXInitialize = init_t(self._init)
        self.ADLXTerminate = term_t(self._term)
        self.versions = []

    # ---- object factory: a struct whose first field points to a vtable of CFUNCTYPE callbacks
    def _obj(self, kind, slots, data=None, refcounted=True):
        n = max([*slots, 2]) + 1
        table = (ctypes.c_void_p * n)()
        if refcounted:
            slots = {1: (ctypes.c_long, [P], self._release), **slots}
        for i, (res, args, fn) in slots.items():
            cb = ctypes.CFUNCTYPE(res, *args)(fn)
            self.keep.append(cb)
            table[i] = ctypes.cast(cb, ctypes.c_void_p)
        obj = ctypes.c_void_p(ctypes.addressof(table))
        box = (ctypes.c_void_p * 1)(obj.value)
        self.keep += [table, box]
        addr = ctypes.addressof(box)
        self.objs[addr] = (kind, data)
        if refcounted:
            self.live.add(addr)
        return addr

    def _new(self, kind, slots, data):
        return self._obj(kind, slots, data)

    def _release(self, this):
        self.live.discard(this)
        return 0

    # ---- exported functions
    def _init(self, version, pp):
        self.versions.append(version)
        if self.init_result == 0:
            pp[0] = self.system
        return self.init_result

    def _term(self):
        self.terminated += 1
        return 0

    # ---- IADLXSystem
    def _get_gpus(self, this, pp):
        pp[0] = self._new("list", {3: (ctypes.c_uint, [P], lambda t: len(self.gpu_objs)),
                                   11: (ctypes.c_int, [P, ctypes.c_uint, PP], self._at)}, None)
        return 0

    def _at(self, this, i, pp):
        pp[0] = self.gpu_objs[i]
        self.live.add(self.gpu_objs[i])        # At acquires a reference the caller must release
        return 0

    def _get_perf(self, this, pp):
        pp[0] = self._new("perf", {18: (ctypes.c_int, [P, P, PP], self._cur), 21: (ctypes.c_int, [P, P, PP], self._sup)}, None)
        return 0

    # ---- IADLXGPU
    def _spec(self, gpu):
        return self.gpus_spec[self.objs[gpu][1]]

    def _name(self, this, out):
        self._name_buf = ctypes.c_char_p(self._spec(this)[0].encode())
        out[0] = self._name_buf.value
        return 0

    def _type(self, this, out):
        out[0] = 1 if self._spec(this)[1] else 2
        return 0

    # ---- IADLXPerformanceMonitoringServices
    def _sup(self, this, gpu, pp):
        _, _, board, gpu_w = self._spec(gpu)
        def flag(v):
            def f(t, out):
                out[0] = 1 if v is not None else 0
                return 0
            return f
        pp[0] = self._new("support", {8: (ctypes.c_int, [P, ctypes.POINTER(ctypes.c_uint8)], flag(gpu_w)),
                                      9: (ctypes.c_int, [P, ctypes.POINTER(ctypes.c_uint8)], flag(board))}, None)
        return 0

    def _cur(self, this, gpu, pp):
        _, _, board, gpu_w = self._spec(gpu)
        def val(v):
            def f(t, out):
                if v is None:
                    return 13  # ADLX_NOT_SUPPORTED
                out[0] = v
                return 0
            return f
        pp[0] = self._new("metrics", {9: (ctypes.c_int, [P, ctypes.POINTER(ctypes.c_double)], val(gpu_w)),
                                      10: (ctypes.c_int, [P, ctypes.POINTER(ctypes.c_double)], val(board))}, None)
        return 0


class Adlx(unittest.TestCase):
    def discover(self, fake, gpus=None):
        notes = []
        with mock.patch.object(meters, "_load_adlx", return_value=fake):
            ms, h = meters.discover_adlx(notes, gpus)
        return ms, h, notes

    def test_discrete_and_builtin_gpus(self):
        fake = FakeAdlx([("AMD Radeon RX 7900 XTX", False, 312.5, 280.0), ("AMD Radeon 8065S", True, None, 84.0)])
        ms, h, notes = self.discover(fake)
        self.assertEqual(fake.versions, [meters.ADLX_VERSION])
        self.assertEqual([m.name for m in ms], ["GPU 0 AMD Radeon RX 7900 XTX", "GPU 1 AMD Radeon 8065S"])
        self.assertEqual([m.reading for m in ms], ["total board power", "GPU power"])
        self.assertEqual([m.method for m in ms], ["adlx power sampling"] * 2)
        self.assertEqual(ms[0].read_w(), 312.5)        # board power preferred when the driver reports it
        self.assertEqual(ms[1].read_w(), 84.0)         # built-in GPU: GPU power
        self.assertTrue(any("built-in GPU" in n for n in notes))
        ms[0].start(0.0); ms[0].poll(2.0)
        self.assertAlmostEqual(ms[0].energy_j, 625.0)  # 312.5 W for 2 s
        h.close()
        self.assertEqual(fake.terminated, 1)
        leftover = {fake.objs[a][0] for a in fake.live}
        self.assertEqual(leftover, set(), "every ADLX object joules got must be released")

    def test_gpu_filter(self):
        fake = FakeAdlx([("AMD Radeon RX 7900 XTX", False, 312.5, None), ("AMD Radeon RX 7600", False, 150.0, None)])
        ms, h, _ = self.discover(fake, gpus=[1])
        self.assertEqual([m.name for m in ms], ["GPU 1 AMD Radeon RX 7600"])
        h.close()

    def test_hip_visible_devices(self):
        fake = FakeAdlx([("AMD Radeon RX 7900 XTX", False, 312.5, None), ("AMD Radeon RX 7600", False, 150.0, None)])
        with mock.patch.dict(meters.os.environ, {"HIP_VISIBLE_DEVICES": "1"}):
            ms, h, _ = self.discover(fake)
        self.assertEqual([m.name for m in ms], ["GPU 1 AMD Radeon RX 7600"])
        h.close()
        with mock.patch.dict(meters.os.environ, {"HIP_VISIBLE_DEVICES": "1"}):
            ms, h, _ = self.discover(FakeAdlx([("A", False, 1.0, None), ("B", False, 2.0, None)]), gpus=[0])
        self.assertEqual([m.name for m in ms], ["GPU 0 A"])   # --gpus wins over the environment
        h.close()

    def test_no_power_reported(self):
        fake = FakeAdlx([("AMD Radeon Pro", False, None, None)])
        ms, h, notes = self.discover(fake)
        self.assertEqual((ms, h), ([], None))
        self.assertTrue(any("reports no power" in n for n in notes))
        self.assertEqual(fake.terminated, 1)

    def test_old_driver(self):
        fake = FakeAdlx([], init_result=5)              # ADLX_BAD_VER
        ms, h, notes = self.discover(fake)
        self.assertEqual((ms, h), ([], None))
        self.assertTrue(any("update the AMD graphics driver" in n for n in notes))

    def test_not_windows(self):
        with mock.patch.object(meters.platform, "system", return_value="Linux"):
            self.assertIsNone(meters._load_adlx())
        notes = []
        with mock.patch.object(meters, "_load_adlx", return_value=None):
            self.assertEqual(meters.discover_adlx(notes), ([], None))
        self.assertEqual(notes, [])


if __name__ == "__main__":
    unittest.main()
