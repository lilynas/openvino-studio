import ctypes
import os
import platform
import re
import threading
import time
import uuid
from collections import deque
from ctypes import wintypes
from typing import Dict, List, Optional

import psutil


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    def __init__(self, guid_str: str):
        u = uuid.UUID(guid_str)
        super().__init__(
            u.time_low,
            u.time_mid,
            u.time_hi_version,
            (ctypes.c_ubyte * 8)(*u.bytes[8:]),
        )


class LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class PDH_FMT_COUNTERVALUE(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("doubleValue", ctypes.c_double)]


class PDH_FMT_COUNTERVALUE_ITEM_W(ctypes.Structure):
    _fields_ = [("szName", wintypes.LPCWSTR), ("FmtValue", PDH_FMT_COUNTERVALUE)]


PDH_FMT_DOUBLE = 0x00000200
PDH_MORE_DATA = 0x800007D2

NPU_KEYWORDS = (
    "npu",
    "ai boost",
    "neural",
    "hexagon",
    "ryzen ai",
    "ipu",
    "xdna",
    "movidius",
    "hailo",
)


def discover_dxcore_adapters() -> Dict[str, dict]:
    """
    Use Windows 11 native DXCore API (dxcore.dll) to enumerate ALL hardware WDDM/MCDM
    adapters — both Display GPUs (Iris Xe, Arc, RTX, Radeon) and Compute-only NPUs
    (Intel AI Boost, AMD Ryzen AI XDNA, Qualcomm Hexagon NPU) — with exact LUIDs.
    """
    adapters: Dict[str, dict] = {}
    if platform.system() != "Windows":
        return adapters

    try:
        dxcore = ctypes.windll.dxcore
        IID_IDXCoreAdapterFactory = GUID("78ee5945-c36e-4b13-a669-005dd11c0f06")
        IID_IDXCoreAdapterList = GUID("526c7776-40e9-459b-b711-f32ad76dfc28")
        IID_IDXCoreAdapter = GUID("f0db4c7f-fe5a-42a2-bd62-f2a6cf6fc83e")

        ATTR_GRAPHICS = GUID("0c9ece4d-2f6e-4f01-8c96-e89e331b47b1")
        ATTR_CORE_COMPUTE = GUID("248e2800-a793-4724-abaa-23a6de1be090")
        ATTR_GENERIC_ML = GUID("b71b0d41-1088-422f-a27c-0250b7d3a988")

        factory = ctypes.c_void_p()
        if dxcore.DXCoreCreateAdapterFactory(ctypes.byref(IID_IDXCoreAdapterFactory), ctypes.byref(factory)) != 0:
            return adapters

        fvt = ctypes.cast(ctypes.cast(factory, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
        CreateAdapterList = ctypes.WINFUNCTYPE(
            ctypes.c_long,
            ctypes.c_void_p,
            wintypes.UINT,
            ctypes.POINTER(GUID),
            ctypes.POINTER(GUID),
            ctypes.POINTER(ctypes.c_void_p),
        )(fvt[3])
        ReleaseFactory = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(fvt[2])

        for attr in (ATTR_GRAPHICS, ATTR_CORE_COMPUTE, ATTR_GENERIC_ML):
            alist = ctypes.c_void_p()
            attrs = (GUID * 1)(attr)
            if CreateAdapterList(factory, 1, attrs, ctypes.byref(IID_IDXCoreAdapterList), ctypes.byref(alist)) != 0 or not alist.value:
                continue

            lvt = ctypes.cast(ctypes.cast(alist, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
            GetAdapter = ctypes.WINFUNCTYPE(
                ctypes.c_long, ctypes.c_void_p, wintypes.UINT, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)
            )(lvt[3])
            GetAdapterCount = ctypes.WINFUNCTYPE(wintypes.UINT, ctypes.c_void_p)(lvt[4])
            ReleaseList = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(lvt[2])

            count = GetAdapterCount(alist)
            for idx in range(count):
                ad = ctypes.c_void_p()
                if GetAdapter(alist, idx, ctypes.byref(IID_IDXCoreAdapter), ctypes.byref(ad)) != 0 or not ad.value:
                    continue
                avt = ctypes.cast(ctypes.cast(ad, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
                IsAttributeSupported = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.POINTER(GUID))(avt[4])
                GetProperty = ctypes.WINFUNCTYPE(
                    ctypes.c_long, ctypes.c_void_p, wintypes.UINT, ctypes.c_size_t, ctypes.c_void_p
                )(avt[6])
                ReleaseAdapter = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(avt[2])

                luid = LUID()
                GetProperty(ad, 0, ctypes.sizeof(luid), ctypes.byref(luid))
                desc_buf = ctypes.create_string_buffer(256)
                GetProperty(ad, 2, 256, desc_buf)
                is_hw = ctypes.c_bool(False)
                GetProperty(ad, 11, ctypes.sizeof(is_hw), ctypes.byref(is_hw))

                # Property 7 = DedicatedAdapterMemory (uint64), 9 = SharedSystemMemory (uint64)
                ded_mem = ctypes.c_uint64(0)
                shr_mem = ctypes.c_uint64(0)
                GetProperty(ad, 7, ctypes.sizeof(ded_mem), ctypes.byref(ded_mem))
                GetProperty(ad, 9, ctypes.sizeof(shr_mem), ctypes.byref(shr_mem))

                is_gfx = bool(IsAttributeSupported(ad, ctypes.byref(ATTR_GRAPHICS)))
                ReleaseAdapter(ad)

                name = desc_buf.value.decode("utf-8", "ignore").strip()
                if not is_hw.value or not name or "Basic Render" in name or "Remote Display" in name:
                    continue

                luid_key = f"0x{luid.HighPart & 0xffffffff:08x}_0x{luid.LowPart:08x}".lower()
                lower_name = name.lower()
                is_npu = (not is_gfx) or any(k in lower_name for k in NPU_KEYWORDS)
                kind = "NPU" if is_npu else "GPU"

                adapters[luid_key] = {
                    "luid": luid_key,
                    "name": name,
                    "kind": kind,
                    "is_graphics": is_gfx,
                    "dedicated_total_mb": round(ded_mem.value / (1024 * 1024), 1),
                    "shared_total_mb": round(shr_mem.value / (1024 * 1024), 1),
                }
            ReleaseList(alist)
        ReleaseFactory(factory)
    except Exception:
        pass

    return adapters


def get_cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                name, _ = winreg.QueryValueEx(k, "ProcessorNameString")
                return name.strip()
        except Exception:
            pass
    return platform.processor() or "Unknown CPU"


LUID_ENGINE_RE = re.compile(
    r"pid_(\d+)_luid_(0x[0-9a-fA-F]+_0x[0-9a-fA-F]+)_phys_\d+_eng_(\d+)_engtype_(.+)$"
)
LUID_ADAPTER_RE = re.compile(r"luid_(0x[0-9a-fA-F]+_0x[0-9a-fA-F]+)_phys_\d+")
LUID_PROC_MEM_RE = re.compile(r"pid_(\d+)_luid_(0x[0-9a-fA-F]+_0x[0-9a-fA-F]+)_phys_\d+")


class HardwareTelemetry:
    """
    Non-blocking background hardware monitor for CPU, System RAM, OVMS process RAM/VRAM,
    GPUs (iGPU + dGPU), and NPUs (MCDM + dedicated NPU performance counters).
    """

    def __init__(self, history_seconds: int = 60):
        self.cpu_name = get_cpu_name()
        self.cpu_cores_physical = psutil.cpu_count(logical=False) or 1
        self.cpu_cores_logical = psutil.cpu_count(logical=True) or 1
        self.adapters = discover_dxcore_adapters()

        self.history = deque(maxlen=history_seconds)
        self.latest: dict = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self._pdh = None
        self._query = wintypes.HANDLE()
        self._counters: Dict[str, wintypes.HANDLE] = {}
        self._init_pdh()

        # Prime psutil cpu_percent
        psutil.cpu_percent(interval=None)
        self._collect_once()

        self._thread = threading.Thread(target=self._loop, name="HardwareTelemetry", daemon=True)
        self._thread.start()

    def _init_pdh(self):
        if platform.system() != "Windows":
            return
        try:
            self._pdh = ctypes.windll.pdh
            if self._pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query)) != 0:
                self._pdh = None
                return

            counter_paths = {
                "gpu_engine": r"\GPU Engine(*)\Utilization Percentage",
                "gpu_ded_mem": r"\GPU Adapter Memory(*)\Dedicated Usage",
                "gpu_shr_mem": r"\GPU Adapter Memory(*)\Shared Usage",
                "gpu_proc_ded": r"\GPU Process Memory(*)\Dedicated Usage",
                "gpu_proc_shr": r"\GPU Process Memory(*)\Shared Usage",
                # Dedicated NPU counter sets on newer Windows 11 24H2+/25H2 Copilot+ builds
                "npu_engine_1": r"\NPU Engine(*)\Utilization Percentage",
                "npu_engine_2": r"\Neural Processing Unit Engine(*)\Utilization Percentage",
            }
            for key, path in counter_paths.items():
                h = wintypes.HANDLE()
                res = self._pdh.PdhAddEnglishCounterW(self._query, path, 0, ctypes.byref(h))
                if res == 0 and h.value:
                    self._counters[key] = h

            self._pdh.PdhCollectQueryData(self._query)
        except Exception:
            self._pdh = None

    def _read_pdh_array(self, key: str) -> List[tuple]:
        if not self._pdh or key not in self._counters:
            return []
        counter = self._counters[key]
        buf_size = wintypes.DWORD(0)
        item_count = wintypes.DWORD(0)
        status = self._pdh.PdhGetFormattedCounterArrayW(
            counter, PDH_FMT_DOUBLE, ctypes.byref(buf_size), ctypes.byref(item_count), None
        )
        if (status & 0xFFFFFFFF) != PDH_MORE_DATA or buf_size.value == 0:
            return []
        buf = (ctypes.c_byte * buf_size.value)()
        status = self._pdh.PdhGetFormattedCounterArrayW(
            counter,
            PDH_FMT_DOUBLE,
            ctypes.byref(buf_size),
            ctypes.byref(item_count),
            ctypes.cast(buf, ctypes.POINTER(PDH_FMT_COUNTERVALUE_ITEM_W)),
        )
        if status != 0:
            return []
        items = ctypes.cast(buf, ctypes.POINTER(PDH_FMT_COUNTERVALUE_ITEM_W * item_count.value)).contents
        out = []
        for it in items:
            if it.szName and it.FmtValue.CStatus == 0:
                out.append((it.szName, float(it.FmtValue.doubleValue)))
        return out

    def _loop(self):
        while not self._stop_event.is_set():
            time.sleep(1.0)
            try:
                self._collect_once()
            except Exception:
                pass

    def _find_ovms_procs(self) -> List[psutil.Process]:
        procs = []
        for p in psutil.process_iter(["pid", "name"]):
            try:
                pname = (p.info.get("name") or "").lower()
                if pname in ("ovms.exe", "ovms"):
                    procs.append(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return procs

    def _collect_once(self):
        now = time.time()
        cpu_pct = round(psutil.cpu_percent(interval=None), 1)
        freq = psutil.cpu_freq()
        cpu_mhz = round(freq.current, 0) if freq else 0

        vm = psutil.virtual_memory()
        ram_total_gb = round(vm.total / (1024 ** 3), 2)
        ram_used_gb = round(vm.used / (1024 ** 3), 2)
        ram_avail_gb = round(vm.available / (1024 ** 3), 2)
        ram_pct = round(vm.percent, 1)

        ovms_procs = self._find_ovms_procs()
        ovms_pids = {p.pid for p in ovms_procs}
        ovms_ram_bytes = 0
        for p in ovms_procs:
            try:
                mem_info = p.memory_info()
                ovms_ram_bytes += getattr(mem_info, "private", mem_info.rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        ovms_ram_gb = round(ovms_ram_bytes / (1024 ** 3), 2)

        # Collect PDH GPU/NPU counters
        if self._pdh:
            try:
                self._pdh.PdhCollectQueryData(self._query)
            except Exception:
                pass

        # Per-LUID engine sums: {luid: {(eng_id, eng_type): util_sum}}
        adapter_eng_totals: Dict[str, Dict[tuple, float]] = {}
        ovms_eng_totals: Dict[str, Dict[tuple, float]] = {}

        for raw_name, val in self._read_pdh_array("gpu_engine"):
            if val <= 0.0:
                continue
            m = LUID_ENGINE_RE.search(raw_name)
            if not m:
                continue
            pid_s, luid_s, eng_id, eng_type = m.groups()
            luid_k = luid_s.lower()
            key = (eng_id, eng_type)
            d = adapter_eng_totals.setdefault(luid_k, {})
            d[key] = d.get(key, 0.0) + val
            if int(pid_s) in ovms_pids:
                od = ovms_eng_totals.setdefault(luid_k, {})
                od[key] = od.get(key, 0.0) + val

        ded_mem_by_luid: Dict[str, float] = {}
        for raw_name, val in self._read_pdh_array("gpu_ded_mem"):
            m = LUID_ADAPTER_RE.search(raw_name)
            if m:
                ded_mem_by_luid[m.group(1).lower()] = val / (1024 * 1024)

        shr_mem_by_luid: Dict[str, float] = {}
        for raw_name, val in self._read_pdh_array("gpu_shr_mem"):
            m = LUID_ADAPTER_RE.search(raw_name)
            if m:
                shr_mem_by_luid[m.group(1).lower()] = val / (1024 * 1024)

        ovms_vram_by_luid: Dict[str, float] = {}
        for counter_key in ("gpu_proc_ded", "gpu_proc_shr"):
            for raw_name, val in self._read_pdh_array(counter_key):
                if val <= 0:
                    continue
                m = LUID_PROC_MEM_RE.search(raw_name)
                if m and int(m.group(1)) in ovms_pids:
                    lk = m.group(2).lower()
                    ovms_vram_by_luid[lk] = ovms_vram_by_luid.get(lk, 0.0) + val / (1024 * 1024)

        gpus = []
        npus = []

        for luid_k, meta in self.adapters.items():
            eng_map = adapter_eng_totals.get(luid_k, {})
            # Windows Task Manager defines overall GPU/NPU utilization as the max across engines
            overall_util = min(100.0, round(max(eng_map.values(), default=0.0), 1))

            # Group by engine type (3D, Compute, Copy, VideoDecode, etc.)
            by_type: Dict[str, float] = {}
            for (_, etype), v in eng_map.items():
                by_type[etype] = max(by_type.get(etype, 0.0), min(100.0, round(v, 1)))

            ovms_map = ovms_eng_totals.get(luid_k, {})
            ovms_util = min(100.0, round(max(ovms_map.values(), default=0.0), 1))

            ded_used = round(ded_mem_by_luid.get(luid_k, 0.0), 1)
            shr_used = round(shr_mem_by_luid.get(luid_k, 0.0), 1)
            ovms_mem_mb = round(ovms_vram_by_luid.get(luid_k, 0.0), 1)

            item = {
                "luid": luid_k,
                "name": meta["name"],
                "kind": meta["kind"],
                "utilization_pct": overall_util,
                "ovms_utilization_pct": ovms_util,
                "engines": by_type,
                "dedicated_used_mb": ded_used,
                "dedicated_total_mb": meta["dedicated_total_mb"],
                "shared_used_mb": shr_used,
                "shared_total_mb": meta["shared_total_mb"],
                "ovms_memory_mb": ovms_mem_mb,
            }
            if meta["kind"] == "NPU":
                npus.append(item)
            else:
                gpus.append(item)

        # Also check standalone NPU performance counter sets if DXCore didn't classify an NPU
        extra_npu_vals = []
        for k in ("npu_engine_1", "npu_engine_2"):
            for _, val in self._read_pdh_array(k):
                if val >= 0:
                    extra_npu_vals.append(val)
        if not npus and extra_npu_vals:
            npus.append({
                "luid": "npu_perf_counter",
                "name": "System Neural Processing Unit (NPU)",
                "kind": "NPU",
                "utilization_pct": min(100.0, round(max(extra_npu_vals, default=0.0), 1)),
                "ovms_utilization_pct": 0.0,
                "engines": {"Compute": min(100.0, round(max(extra_npu_vals, default=0.0), 1))},
                "dedicated_used_mb": 0.0,
                "dedicated_total_mb": 0.0,
                "shared_used_mb": 0.0,
                "shared_total_mb": 0.0,
                "ovms_memory_mb": 0.0,
            })

        primary_gpu_pct = max((g["utilization_pct"] for g in gpus), default=0.0)
        primary_npu_pct = max((n["utilization_pct"] for n in npus), default=0.0) if npus else None

        snapshot = {
            "timestamp": round(now, 1),
            "time_str": time.strftime("%H:%M:%S", time.localtime(now)),
            "cpu": {
                "name": self.cpu_name,
                "cores_physical": self.cpu_cores_physical,
                "cores_logical": self.cpu_cores_logical,
                "utilization_pct": cpu_pct,
                "freq_mhz": cpu_mhz,
            },
            "memory": {
                "total_gb": ram_total_gb,
                "used_gb": ram_used_gb,
                "available_gb": ram_avail_gb,
                "utilization_pct": ram_pct,
                "ovms_ram_gb": ovms_ram_gb,
                "ovms_pids": sorted(ovms_pids),
            },
            "gpus": gpus,
            "npus": npus,
            "npu_present": len(npus) > 0,
            "summary": {
                "cpu_pct": cpu_pct,
                "ram_pct": ram_pct,
                "ovms_ram_gb": ovms_ram_gb,
                "gpu_pct": primary_gpu_pct,
                "npu_pct": primary_npu_pct,
            },
        }

        history_point = {
            "t": snapshot["time_str"],
            "cpu": cpu_pct,
            "ram": ram_pct,
            "ovms_ram_gb": ovms_ram_gb,
            "gpu": primary_gpu_pct,
            "npu": primary_npu_pct if primary_npu_pct is not None else 0.0,
        }

        with self._lock:
            self.latest = snapshot
            self.history.append(history_point)

    def get_telemetry(self) -> dict:
        with self._lock:
            return {
                "current": self.latest,
                "history": list(self.history),
            }

    def close(self):
        self._stop_event.set()
        if self._pdh and self._query.value:
            try:
                self._pdh.PdhCloseQuery(self._query)
            except Exception:
                pass
