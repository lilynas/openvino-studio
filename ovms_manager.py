import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil


WEEKLY_VERSION_URL = "https://storage.openvinotoolkit.org/repositories/openvino_model_server/packages/weekly/latest/version.txt"
WEEKLY_ZIP_URL = "https://storage.openvinotoolkit.org/repositories/openvino_model_server/packages/weekly/latest/ovms_windows.zip"
WEEKLY_SHA256_URL = "https://storage.openvinotoolkit.org/repositories/openvino_model_server/packages/weekly/latest/ovms_windows.zip.sha256"
GITHUB_LATEST_RELEASE_API = "https://api.github.com/repos/openvinotoolkit/model_server/releases/latest"


STUDIO_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(STUDIO_DIR, "studio_state.json")

_PARENT_DIR = os.path.dirname(STUDIO_DIR)
DEFAULT_MODELS_ROOT = os.environ.get(
    "OVMS_MODELS_ROOT",
    _PARENT_DIR if os.path.basename(_PARENT_DIR).lower() == "llms" else os.path.join(STUDIO_DIR, "models"),
)
DEFAULT_OVMS_DIR = os.environ.get(
    "OVMS_DIR",
    os.path.join(os.path.expanduser("~"), ".cherrystudio", "ovms", "ovms"),
)
DEFAULT_CONFIG_JSON = os.environ.get(
    "OVMS_CONFIG_JSON",
    os.path.join(DEFAULT_OVMS_DIR, "models", "config.json"),
)
DEFAULT_CACHE_DIR = os.path.join(DEFAULT_MODELS_ROOT, "ovms_cache")
STDOUT_LOG = os.path.join(DEFAULT_MODELS_ROOT, "ovms_stdout.log")
STDERR_LOG = os.path.join(DEFAULT_MODELS_ROOT, "ovms_stderr.log")

RECOMMENDED_CATALOG = [
    {
        "repo_id": "OpenVINO/Qwen3-4B-int4-ov",
        "name": "Qwen3-4B-int4-ov",
        "title": "Qwen3 4B (INT4 极速版)",
        "size_gb": 2.13,
        "ram_est_gb": 3.2,
        "category": "LLM / 轻量思考",
        "desc": "阿里最新一代 Qwen3 4B 预量化模型，支持原生 Thinking 思考模式，核显秒回，仅占 3GB 内存。",
        "recommended_device": "GPU",
    },
    {
        "repo_id": "OpenVINO/Qwen3-8B-int4-ov",
        "name": "Qwen3-8B-int4-ov",
        "title": "Qwen3 8B (INT4 主力均衡版)",
        "size_gb": 4.55,
        "ram_est_gb": 6.0,
        "category": "LLM / 全能主力",
        "desc": "8B 黄金尺寸，兼顾深度推理与日常代码编写，在 16GB RAM 预算内游刃有余。",
        "recommended_device": "GPU",
    },
    {
        "repo_id": "OpenVINO/Qwen3-14B-int4-ov",
        "name": "Qwen3-14B-int4-ov",
        "title": "Qwen3 14B (INT4 旗舰智商版)",
        "size_gb": 9.06,
        "ram_est_gb": 11.2,
        "category": "LLM / 深度逻辑",
        "desc": "16GB 内存红线下的中文常识与复杂推理天花板，适合严肃写作、长文档总结与代码架构。",
        "recommended_device": "GPU",
    },
    {
        "repo_id": "OpenVINO/gemma-4-E4B-it-int4-ov",
        "name": "gemma-4-E4B-it-int4-ov",
        "title": "Gemma 4 E4B (INT4 多模态视觉)",
        "size_gb": 6.09,
        "ram_est_gb": 7.5,
        "category": "VLM / 多模态视觉",
        "desc": "Google 最新第四代多模态模型（需 OVMS 2026.5+），支持图文理解与工具调用。",
        "recommended_device": "GPU",
    },
    {
        "repo_id": "OpenVINO/gemma-4-12B-it-int4-ov",
        "name": "gemma-4-12B-it-int4-ov",
        "title": "Gemma 4 12B (INT4 旗舰多模态)",
        "size_gb": 7.45,
        "ram_est_gb": 9.5,
        "category": "VLM / 多模态旗舰",
        "desc": "Gemma 4 12B INT4 视觉语言全能版，代码与多语言能力极强，完美契合 16GB 内存预算。",
        "recommended_device": "GPU",
    },
    {
        "repo_id": "OpenVINO/Phi-4-mini-instruct-int4-ov",
        "name": "Phi-4-mini-instruct-int4-ov",
        "title": "Phi-4 Mini 3.8B (INT4 理科小钢炮)",
        "size_gb": 2.22,
        "ram_est_gb": 3.4,
        "category": "LLM / 极速轻量",
        "desc": "微软 Phi-4 浓缩精华版，逻辑与数学密度极高，适合低功耗后台常驻。",
        "recommended_device": "GPU",
    },
]


def _norm_path(p: str) -> str:
    return os.path.normpath(p).replace("\\", "/")


def parse_graph_pbtxt(pbtxt_path: str) -> Dict[str, Any]:
    """Parse key LLMCalculatorOptions from a model's graph.pbtxt."""
    defaults = {
        "device": "GPU",
        "models_path": "./",
        "kv_cache_precision": "f16",
        "max_num_seqs": 256,
        "cache_size": 0,
        "enable_prefix_caching": True,
        "dynamic_split_fuse": True,
        "pipeline_type": "AUTO",
        "performance_hint": "",
        "num_streams": "",
        "max_prompt_len": 0,
        "reasoning_parser": "",
        "tool_parser": "",
        "extra_plugin_config": {},
    }
    if not os.path.exists(pbtxt_path):
        return defaults

    try:
        text = Path(pbtxt_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return defaults

    def _extract_str(key: str, fallback: str = "") -> str:
        m = re.search(rf'\b{key}\s*:\s*["\']([^"\']*)["\']', text)
        return m.group(1).strip() if m else fallback

    def _extract_int(key: str, fallback: int = 0) -> int:
        m = re.search(rf"\b{key}\s*:\s*(\d+)", text)
        return int(m.group(1)) if m else fallback

    def _extract_bool(key: str, fallback: bool = True) -> bool:
        m = re.search(rf"\b{key}\s*:\s*(true|false)", text, re.IGNORECASE)
        return (m.group(1).lower() == "true") if m else fallback

    defaults["device"] = _extract_str("device", "GPU")
    defaults["models_path"] = _extract_str("models_path", "./")
    defaults["max_num_seqs"] = _extract_int("max_num_seqs", 256)
    defaults["cache_size"] = _extract_int("cache_size", 0)
    defaults["enable_prefix_caching"] = _extract_bool("enable_prefix_caching", True)
    defaults["dynamic_split_fuse"] = _extract_bool("dynamic_split_fuse", True)
    defaults["pipeline_type"] = _extract_str("pipeline_type", "AUTO")
    defaults["reasoning_parser"] = _extract_str("reasoning_parser", "")
    defaults["tool_parser"] = _extract_str("tool_parser", "")

    # Extract plugin_config JSON string
    m_pc = re.search(r"plugin_config\s*:\s*'([^']*)'", text)
    if not m_pc:
        m_pc = re.search(r'plugin_config\s*:\s*"((?:\\.|[^"\\])*)"', text)
        raw_pc = m_pc.group(1).replace('\\"', '"') if m_pc else ""
    else:
        raw_pc = m_pc.group(1)

    if raw_pc:
        try:
            pc_dict = json.loads(raw_pc)
            if isinstance(pc_dict, dict):
                defaults["kv_cache_precision"] = pc_dict.pop("KV_CACHE_PRECISION", "")
                defaults["performance_hint"] = pc_dict.pop("PERFORMANCE_HINT", "")
                defaults["num_streams"] = str(pc_dict.pop("NUM_STREAMS", ""))
                mpl = pc_dict.pop("MAX_PROMPT_LEN", 0)
                defaults["max_prompt_len"] = int(mpl) if str(mpl).isdigit() else 0
                defaults["extra_plugin_config"] = pc_dict
        except Exception:
            pass

    return defaults


def build_graph_pbtxt(params: Dict[str, Any]) -> str:
    """Build a valid OVMS 2026.5 MediaPipe LLM/VLM graph.pbtxt from structured parameters."""
    device = (params.get("device") or "GPU").strip()
    models_path = (params.get("models_path") or "./").strip()
    max_num_seqs = int(params.get("max_num_seqs") or 256)
    cache_size = int(params.get("cache_size") or 0)
    enable_prefix_caching = bool(params.get("enable_prefix_caching", True))
    dynamic_split_fuse = bool(params.get("dynamic_split_fuse", True))
    pipeline_type = (params.get("pipeline_type") or "AUTO").strip()
    reasoning_parser = (params.get("reasoning_parser") or "").strip()
    tool_parser = (params.get("tool_parser") or "").strip()

    plugin_cfg: Dict[str, Any] = {}
    extra = params.get("extra_plugin_config")
    if isinstance(extra, dict):
        plugin_cfg.update(extra)

    kv_prec = (params.get("kv_cache_precision") or "").strip()
    if kv_prec and kv_prec != "default":
        plugin_cfg["KV_CACHE_PRECISION"] = kv_prec

    perf_hint = (params.get("performance_hint") or "").strip()
    if perf_hint and perf_hint != "default":
        plugin_cfg["PERFORMANCE_HINT"] = perf_hint

    num_streams = str(params.get("num_streams") or "").strip()
    if num_streams:
        plugin_cfg["NUM_STREAMS"] = num_streams

    max_prompt_len = int(params.get("max_prompt_len") or 0)
    if max_prompt_len > 0:
        plugin_cfg["MAX_PROMPT_LEN"] = max_prompt_len

    opts_lines = [
        f'      models_path: "{models_path}",',
        f'      device: "{device}",',
        f"      max_num_seqs: {max_num_seqs},",
        f"      enable_prefix_caching: {'true' if enable_prefix_caching else 'false'},",
        f"      dynamic_split_fuse: {'true' if dynamic_split_fuse else 'false'},",
    ]
    if cache_size > 0:
        opts_lines.append(f"      cache_size: {cache_size},")
    if pipeline_type and pipeline_type != "AUTO":
        opts_lines.append(f'      pipeline_type: "{pipeline_type}",')
    if reasoning_parser:
        opts_lines.append(f'      reasoning_parser: "{reasoning_parser}",')
    if tool_parser:
        opts_lines.append(f'      tool_parser: "{tool_parser}",')
    if plugin_cfg:
        pc_json = json.dumps(plugin_cfg, ensure_ascii=False, separators=(",", ":"))
        opts_lines.append(f"      plugin_config: '{pc_json}'")

    opts_block = "\n".join(opts_lines)
    return f"""input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"
node: {{
  name: "LLMExecutor"
  calculator: "HttpLLMCalculator"
  input_stream: "LOOPBACK:loopback"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  input_side_packet: "LLM_NODE_RESOURCES:llm"
  output_stream: "LOOPBACK:loopback"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  input_stream_info: {{
    tag_index: 'LOOPBACK:0',
    back_edge: true
  }}
  node_options: {{
    [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {{
{opts_block}
    }}
  }}
  input_stream_handler {{
    input_stream_handler: "SyncSetInputStreamHandler",
    options {{
      [mediapipe.SyncSetInputStreamHandlerOptions.ext] {{
        sync_set {{
          tag_index: "LOOPBACK:0"
        }}
      }}
    }}
  }}
}}
"""


def inspect_model_directory(model_dir: str) -> Dict[str, Any]:
    """Inspect a local directory to determine if it is a valid OpenVINO IR / GGUF model and extract metadata."""
    p = Path(model_dir)
    if not p.is_dir():
        return {"valid": False}

    files = [f.name for f in p.iterdir() if f.is_file()]
    lower_files = {f.lower() for f in files}
    gguf_files = [f for f in files if f.lower().endswith(".gguf")]

    has_ov_llm = "openvino_language_model.xml" in lower_files or "openvino_model.xml" in lower_files
    has_vlm = "openvino_vision_embeddings_model.xml" in lower_files or "openvino_text_embeddings_model.xml" in lower_files
    has_pbtxt = "graph.pbtxt" in lower_files

    if not (has_ov_llm or gguf_files or has_pbtxt):
        return {"valid": False}

    total_bytes = 0
    for root, _, fnames in os.walk(model_dir):
        for fn in fnames:
            try:
                total_bytes += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass

    size_gb = round(total_bytes / (1024 ** 3), 2)
    task_type = "VLM (多模态视觉)" if has_vlm else ("GGUF 模型" if gguf_files and not has_ov_llm else "LLM (大语言模型)")

    arch = ""
    quant = "INT4" if "int4" in p.name.lower() else ("INT8" if "int8" in p.name.lower() else ("FP16" if "fp16" in p.name.lower() else "OV-IR"))
    cfg_file = p / "config.json"
    if cfg_file.exists():
        try:
            cfg_data = json.loads(cfg_file.read_text(encoding="utf-8", errors="ignore"))
            arch = cfg_data.get("model_type") or (cfg_data.get("architectures") or [""])[0]
        except Exception:
            pass

    pbtxt_path = os.path.join(model_dir, "graph.pbtxt")
    params = parse_graph_pbtxt(pbtxt_path)

    return {
        "valid": True,
        "name": p.name,
        "base_path": _norm_path(str(p.resolve())),
        "size_gb": size_gb,
        "task_type": task_type,
        "architecture": arch,
        "quantization": quant,
        "has_pbtxt": os.path.exists(pbtxt_path),
        "gguf_files": gguf_files,
        "params": params,
    }


class OvmsManager:
    def __init__(self):
        os.makedirs(STUDIO_DIR, exist_ok=True)
        os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
        self._lock = threading.Lock()
        self.state = self._load_state()
        self.last_active_ts = time.time()
        self.downloads: Dict[str, Dict[str, Any]] = {}
        self._installed_version_cache: Optional[Dict[str, str]] = None
        self.updater_state: Dict[str, Any] = {
            "checking": False,
            "updating": False,
            "phase": "idle",
            "progress_pct": 0.0,
            "downloaded_mb": 0.0,
            "total_mb": 0.0,
            "speed_mbps": 0.0,
            "message": "",
            "error": "",
            "last_checked_ts": 0.0,
            "update_available": False,
            "installed": {},
            "remote": {},
            "last_updated_ts": 0.0,
        }
        self.discover_and_sync_models(write_ovms_config=False)

        # Refresh local installed OVMS version immediately
        self.updater_state["installed"] = self.get_installed_ovms_version(force=True)

        # Background idle-unload & auto-update check thread
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, name="OvmsWatchdog", daemon=True)
        self._watchdog_thread.start()

        # Perform initial non-blocking version check on startup if auto_check_updates is enabled
        if self.state["settings"].get("auto_check_updates", True):
            threading.Thread(target=self._initial_update_check, name="OvmsInitUpdateCheck", daemon=True).start()

    def _default_state(self) -> Dict[str, Any]:
        return {
            "settings": {
                "ovms_dir": DEFAULT_OVMS_DIR,
                "config_json_path": DEFAULT_CONFIG_JSON,
                "models_root": DEFAULT_MODELS_ROOT,
                "cache_dir": DEFAULT_CACHE_DIR,
                "rest_bind_address": "127.0.0.1",
                "rest_port": 8000,
                "grpc_bind_address": "127.0.0.1",
                "grpc_port": 9000,
                "log_level": "INFO",
                "file_system_poll_wait_seconds": 1,
                "enable_cache_dir": True,
                "shared_vram_gb": 16.0,       # 自定义共享显存大小 / 显存预算上限 (默认 16.0 GB)
                "ram_budget_gb": 16.0,        # 模型安全内存预算线 (默认与 shared_vram_gb 同步为 16.0 GB)
                "idle_auto_stop_minutes": 0,  # 0 = disabled
                "update_channel": "weekly",   # "weekly" (前沿版支持 Gemma4 等新模型) | "stable" (GitHub 正式版)
                "auto_check_updates": True,   # 启动时及后台定期检测新版本
                "check_interval_hours": 6,    # 自动检测间隔（小时）
                "auto_apply_updates": False,  # 检测到新版本后是否自动下载并热替换升级（默认手动确认）
            },
            "models": {},
            "excluded_paths": [],
        }

    def _load_state(self) -> Dict[str, Any]:
        state = self._default_state()
        if os.path.exists(STATE_FILE):
            try:
                saved = json.loads(Path(STATE_FILE).read_text(encoding="utf-8"))
                if "settings" in saved:
                    state["settings"].update(saved["settings"])
                if "models" in saved and isinstance(saved["models"], dict):
                    state["models"] = saved["models"]
                if "excluded_paths" in saved and isinstance(saved["excluded_paths"], list):
                    state["excluded_paths"] = [_norm_path(p) for p in saved["excluded_paths"] if p]
            except Exception:
                pass
        return state

    def _save_state(self):
        try:
            Path(STATE_FILE).write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _read_ovms_config_json(self) -> List[Dict[str, str]]:
        cfg_path = self.state["settings"]["config_json_path"]
        if not os.path.exists(cfg_path):
            return []
        try:
            data = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
            return data.get("mediapipe_config_list", [])
        except Exception:
            return []

    def _write_ovms_config_json(self):
        cfg_path = self.state["settings"]["config_json_path"]
        os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
        mp_list = []
        for mname, mentry in self.state["models"].items():
            if mentry.get("enabled", True) and os.path.isdir(mentry.get("base_path", "")):
                # Ensure graph.pbtxt exists before enabling in OVMS
                pb_path = os.path.join(mentry["base_path"], "graph.pbtxt")
                if not os.path.exists(pb_path):
                    Path(pb_path).write_text(build_graph_pbtxt(mentry.get("params") or {}), encoding="utf-8")
                mp_list.append({
                    "name": mname,
                    "base_path": _norm_path(mentry["base_path"]),
                })
        payload = {
            "mediapipe_config_list": mp_list,
            "model_config_list": [],
        }
        Path(cfg_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def discover_and_sync_models(self, write_ovms_config: bool = True, clear_excluded: bool = False) -> List[Dict[str, Any]]:
        with self._lock:
            if clear_excluded:
                self.state["excluded_paths"] = []
            excluded_set = {_norm_path(p).lower() for p in self.state.get("excluded_paths", []) if p}

            ovms_mp_list = self._read_ovms_config_json()
            enabled_in_ovms = {item.get("name"): _norm_path(item.get("base_path", "")) for item in ovms_mp_list if item.get("name")}

            # Create path->existing_entry map so custom model names & locations are preserved
            path_to_existing: Dict[str, Dict[str, Any]] = {}
            for mname, mentry in list(self.state["models"].items()):
                bp = mentry.get("base_path", "")
                if bp and os.path.isdir(bp):
                    norm_bp = _norm_path(bp)
                    if norm_bp.lower() not in excluded_set:
                        path_to_existing[norm_bp.lower()] = mentry

            # Scan directories
            candidate_dirs = set()
            for _, bp in enabled_in_ovms.items():
                if os.path.isdir(bp) and _norm_path(bp).lower() not in excluded_set:
                    candidate_dirs.add(_norm_path(bp))

            for mentry in list(self.state["models"].values()):
                bp = mentry.get("base_path", "")
                if bp and os.path.isdir(bp) and _norm_path(bp).lower() not in excluded_set:
                    candidate_dirs.add(_norm_path(bp))

            roots_to_scan = [
                self.state["settings"]["models_root"],
                os.path.dirname(self.state["settings"]["config_json_path"]),
            ]
            for root in roots_to_scan:
                if root and os.path.isdir(root):
                    for child in os.listdir(root):
                        if child.lower() in ("ovms-studio", "ovms_cache", "ovms_weekly_extracted", "__pycache__"):
                            continue
                        full = os.path.join(root, child)
                        if os.path.isdir(full):
                            nfull = _norm_path(full)
                            if nfull.lower() not in excluded_set:
                                candidate_dirs.add(nfull)

            updated_models: Dict[str, Dict[str, Any]] = {}
            for cdir in sorted(candidate_dirs):
                info = inspect_model_directory(cdir)
                if not info.get("valid"):
                    continue
                existing = path_to_existing.get(_norm_path(cdir).lower())
                mname = existing["name"] if (existing and existing.get("name")) else info["name"]
                prev = self.state["models"].get(mname, existing or {})
                # Determine enabled state: if previously recorded, keep it; else check if in config.json
                if "enabled" in prev:
                    enabled = bool(prev["enabled"])
                else:
                    enabled = mname in enabled_in_ovms or (_norm_path(cdir) in enabled_in_ovms.values())

                updated_models[mname] = {
                    "name": mname,
                    "base_path": info["base_path"],
                    "enabled": enabled,
                    "size_gb": info["size_gb"],
                    "task_type": info["task_type"],
                    "architecture": info["architecture"],
                    "quantization": info["quantization"],
                    "has_pbtxt": info["has_pbtxt"],
                    "params": info["params"],
                }

            self.state["models"] = updated_models
            self._save_state()
            if write_ovms_config:
                self._write_ovms_config_json()

        return self.get_models_with_live_status()

    def _get_lan_ip(self) -> str:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.settimeout(0.2)
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                if ip and not ip.startswith("127."):
                    return ip
        except Exception:
            pass
        try:
            ip = socket.gethostbyname(socket.gethostname())
            if ip and not ip.startswith("127."):
                return ip
        except Exception:
            pass
        return "127.0.0.1"

    def get_probe_host(self) -> str:
        bind_addr = str(self.state["settings"].get("rest_bind_address") or "127.0.0.1").strip()
        if not bind_addr or bind_addr in ("0.0.0.0", "::", "*", "localhost"):
            return "127.0.0.1"
        return bind_addr

    def get_live_ovms_states(self) -> Dict[str, Dict[str, Any]]:
        """Query http://{probe_host}:{rest_port}/v1/config to get real-time OVMS servable state."""
        host = self.get_probe_host()
        port = self.state["settings"]["rest_port"]
        url = f"http://{host}:{port}/v1/config"
        states: Dict[str, Dict[str, Any]] = {}
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=1.2) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
                for mname, mval in data.items():
                    mv_list = mval.get("model_version_status", [])
                    if mv_list:
                        st = mv_list[0].get("state", "UNKNOWN")
                        err = mv_list[0].get("status", {}).get("error_code", "OK")
                        msg = mv_list[0].get("status", {}).get("error_message", "")
                        states[mname] = {"state": st, "error_code": err, "error_message": msg}
        except Exception:
            pass
        return states

    def trigger_ovms_config_reload(self) -> Dict[str, Any]:
        """Call POST /v1/config/reload so OVMS immediately loads/unloads changed models."""
        host = self.get_probe_host()
        port = self.state["settings"]["rest_port"]
        url = f"http://{host}:{port}/v1/config/reload"
        try:
            req = urllib.request.Request(url, data=b"", method="POST")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return {"reloaded": True, "body": json.loads(resp.read().decode("utf-8", errors="ignore"))}
        except Exception as e:
            return {"reloaded": False, "error": str(e)}

    def get_models_with_live_status(self) -> List[Dict[str, Any]]:
        live_map = self.get_live_ovms_states()
        running_info = self.get_server_status()
        is_running = running_info["running"]
        out = []
        for mname, item in self.state["models"].items():
            copy_item = dict(item)
            if not is_running:
                copy_item["runtime_state"] = "STOPPED"
                copy_item["runtime_detail"] = "服务未启动"
            elif mname in live_map:
                copy_item["runtime_state"] = live_map[mname]["state"]
                copy_item["runtime_detail"] = live_map[mname]["error_message"] or live_map[mname]["error_code"]
            elif copy_item.get("enabled"):
                copy_item["runtime_state"] = "LOADING"
                copy_item["runtime_detail"] = "正在编译加载或等待重载"
            else:
                copy_item["runtime_state"] = "DISABLED"
                copy_item["runtime_detail"] = "已停用（不占内存）"
            out.append(copy_item)
        return out

    def get_server_status(self) -> Dict[str, Any]:
        procs = []
        for p in psutil.process_iter(["pid", "name", "create_time", "cmdline"]):
            try:
                if (p.info.get("name") or "").lower() in ("ovms.exe", "ovms"):
                    procs.append(p)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        running = len(procs) > 0
        pids = [p.pid for p in procs]
        uptime_sec = 0
        if procs:
            try:
                uptime_sec = int(max(0.0, time.time() - min(p.create_time() for p in procs)))
            except Exception:
                pass

        live_states = self.get_live_ovms_states() if running else {}
        api_ready = len(live_states) > 0 or (running and self._probe_rest_alive())

        # Compute shader cache folder size
        cache_dir = self.state["settings"].get("cache_dir", DEFAULT_CACHE_DIR)
        cache_mb = 0.0
        if os.path.isdir(cache_dir):
            try:
                total_c = sum(
                    os.path.getsize(os.path.join(r, f))
                    for r, _, fs in os.walk(cache_dir)
                    for f in fs
                )
                cache_mb = round(total_c / (1024 * 1024), 1)
            except Exception:
                pass

        rest_bind = str(self.state["settings"].get("rest_bind_address") or "127.0.0.1").strip()
        rest_port = int(self.state["settings"].get("rest_port") or 8000)
        grpc_bind = str(self.state["settings"].get("grpc_bind_address") or rest_bind or "127.0.0.1").strip()
        grpc_port = int(self.state["settings"].get("grpc_port") or 9000)
        lan_ip = self._get_lan_ip()
        display_host = "127.0.0.1" if rest_bind in ("0.0.0.0", "::", "*") else rest_bind

        return {
            "running": running,
            "api_ready": api_ready,
            "pids": pids,
            "uptime_seconds": uptime_sec,
            "rest_bind_address": rest_bind,
            "rest_port": rest_port,
            "grpc_bind_address": grpc_bind,
            "grpc_port": grpc_port,
            "lan_ip": lan_ip,
            "endpoint_v1": f"http://{display_host}:{rest_port}/v1",
            "endpoint_v3": f"http://{display_host}:{rest_port}/v3",
            "lan_endpoint_v3": f"http://{lan_ip}:{rest_port}/v3",
            "bind_endpoint_v3": f"http://{rest_bind}:{rest_port}/v3",
            "cache_dir": cache_dir,
            "cache_size_mb": cache_mb,
            "live_models": live_states,
            "settings": self.state["settings"],
            "updater": self.get_updater_status(),
        }

    def _probe_rest_alive(self) -> bool:
        host = self.get_probe_host()
        port = self.state["settings"]["rest_port"]
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/v1/config", timeout=0.8) as r:
                return r.status == 200
        except Exception:
            return False

    def start_server(self) -> Dict[str, Any]:
        status = self.get_server_status()
        if status["running"]:
            return {"ok": True, "message": "OVMS 服务已在运行中", "status": status}

        self._write_ovms_config_json()
        settings = self.state["settings"]
        ovms_dir = settings["ovms_dir"]
        ovms_exe = os.path.join(ovms_dir, "ovms.exe")
        if not os.path.exists(ovms_exe):
            return {"ok": False, "error": f"找不到 ovms.exe: {ovms_exe}"}

        env = os.environ.copy()
        py_home = os.path.join(ovms_dir, "python")
        env["PYTHONHOME"] = py_home
        env["PATH"] = f"{ovms_dir};{py_home};{py_home}\\Scripts;" + env.get("PATH", "")

        rest_bind = str(settings.get("rest_bind_address") or "127.0.0.1").strip()
        grpc_bind = str(settings.get("grpc_bind_address") or rest_bind or "127.0.0.1").strip()

        cmd = [
            ovms_exe,
            "--config_path",
            settings["config_json_path"],
            "--port",
            str(settings["grpc_port"]),
            "--grpc_bind_address",
            grpc_bind,
            "--rest_port",
            str(settings["rest_port"]),
            "--rest_bind_address",
            rest_bind,
            "--log_level",
            settings.get("log_level", "INFO"),
            "--file_system_poll_wait_seconds",
            str(settings.get("file_system_poll_wait_seconds", 1)),
        ]
        if settings.get("enable_cache_dir", True) and settings.get("cache_dir"):
            os.makedirs(settings["cache_dir"], exist_ok=True)
            cmd.extend(["--cache_dir", settings["cache_dir"]])

        stdout_f = open(STDOUT_LOG, "a", encoding="utf-8", errors="ignore")
        stderr_f = open(STDERR_LOG, "a", encoding="utf-8", errors="ignore")
        stdout_f.write(f"\n--- [OVMS Studio Start at {time.strftime('%Y-%m-%d %H:%M:%S')}] CMD: {' '.join(cmd)} ---\n")
        stdout_f.flush()

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) if os.name == "nt" else 0
        proc = subprocess.Popen(
            cmd,
            cwd=ovms_dir,
            env=env,
            stdout=stdout_f,
            stderr=stderr_f,
            creationflags=creationflags,
        )
        self.last_active_ts = time.time()
        time.sleep(0.8)
        return {"ok": True, "pid": proc.pid, "message": "OpenVINO 服务已启动", "status": self.get_server_status()}

    def stop_server(self) -> Dict[str, Any]:
        killed = []
        for p in psutil.process_iter(["pid", "name"]):
            try:
                if (p.info.get("name") or "").lower() in ("ovms.exe", "ovms"):
                    killed.append(p.pid)
                    p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if killed:
            time.sleep(0.5)
        return {
            "ok": True,
            "killed_pids": killed,
            "message": f"已停止 OVMS 服务并释放核显/系统内存 (PID: {killed})" if killed else "服务当前未运行",
            "status": self.get_server_status(),
        }

    def restart_server(self) -> Dict[str, Any]:
        self.stop_server()
        time.sleep(0.4)
        return self.start_server()

    def set_model_enabled(self, model_name: str, enabled: bool) -> Dict[str, Any]:
        with self._lock:
            if model_name not in self.state["models"]:
                return {"ok": False, "error": f"未找到模型: {model_name}"}
            self.state["models"][model_name]["enabled"] = bool(enabled)
            self._save_state()
            self._write_ovms_config_json()

        reload_res = self.trigger_ovms_config_reload()
        return {
            "ok": True,
            "model_name": model_name,
            "enabled": bool(enabled),
            "reload": reload_res,
            "models": self.get_models_with_live_status(),
        }

    def update_model_params(
        self, model_name: str, params: Dict[str, Any], raw_pbtxt: Optional[str] = None
    ) -> Dict[str, Any]:
        with self._lock:
            if model_name not in self.state["models"]:
                return {"ok": False, "error": f"未找到模型: {model_name}"}
            entry = self.state["models"][model_name]
            base_path = entry["base_path"]
            pbtxt_path = os.path.join(base_path, "graph.pbtxt")

            if raw_pbtxt is not None and raw_pbtxt.strip():
                content = raw_pbtxt
            else:
                merged = dict(entry.get("params") or {})
                merged.update(params)
                content = build_graph_pbtxt(merged)

            Path(pbtxt_path).write_text(content, encoding="utf-8")
            entry["params"] = parse_graph_pbtxt(pbtxt_path)
            entry["has_pbtxt"] = True
            self._save_state()

        reload_res = self.trigger_ovms_config_reload()
        return {
            "ok": True,
            "model_name": model_name,
            "params": entry["params"],
            "pbtxt": content,
            "reload": reload_res,
        }

    def get_model_pbtxt(self, model_name: str) -> Dict[str, Any]:
        if model_name not in self.state["models"]:
            return {"ok": False, "error": f"未找到模型: {model_name}"}
        entry = self.state["models"][model_name]
        pbtxt_path = os.path.join(entry["base_path"], "graph.pbtxt")
        raw = Path(pbtxt_path).read_text(encoding="utf-8", errors="ignore") if os.path.exists(pbtxt_path) else build_graph_pbtxt(entry.get("params") or {})
        return {
            "ok": True,
            "model_name": model_name,
            "pbtxt_path": _norm_path(pbtxt_path),
            "pbtxt": raw,
            "params": parse_graph_pbtxt(pbtxt_path) if os.path.exists(pbtxt_path) else entry.get("params", {}),
        }

    def import_local_model(self, folder_path: str, custom_name: Optional[str] = None, enable: bool = True, default_device: str = "GPU") -> Dict[str, Any]:
        folder_path = os.path.abspath(folder_path.strip().strip('"'))
        if os.path.isfile(folder_path) and folder_path.lower().endswith(".gguf"):
            # If user selected a single .gguf file, use its parent directory
            folder_path = os.path.dirname(folder_path)

        info = inspect_model_directory(folder_path)
        if not info.get("valid"):
            return {
                "ok": False,
                "error": f"目录 '{folder_path}' 中未检测到有效的 OpenVINO IR (.xml/.bin) 或 .gguf 模型文件",
            }

        mname = (custom_name or info["name"]).strip()
        pbtxt_path = os.path.join(folder_path, "graph.pbtxt")
        if not os.path.exists(pbtxt_path):
            init_params = info.get("params") or {}
            init_params["device"] = default_device
            init_params["kv_cache_precision"] = "f16"
            Path(pbtxt_path).write_text(build_graph_pbtxt(init_params), encoding="utf-8")
            info = inspect_model_directory(folder_path)

        with self._lock:
            norm_bp = _norm_path(info["base_path"])
            if "excluded_paths" in self.state:
                self.state["excluded_paths"] = [
                    p for p in self.state["excluded_paths"] if _norm_path(p).lower() != norm_bp.lower()
                ]
            self.state["models"][mname] = {
                "name": mname,
                "base_path": info["base_path"],
                "enabled": bool(enable),
                "size_gb": info["size_gb"],
                "task_type": info["task_type"],
                "architecture": info["architecture"],
                "quantization": info["quantization"],
                "has_pbtxt": True,
                "params": info["params"],
            }
            self._save_state()
            self._write_ovms_config_json()

        reload_res = self.trigger_ovms_config_reload()
        return {
            "ok": True,
            "model": self.state["models"][mname],
            "reload": reload_res,
            "models": self.get_models_with_live_status(),
        }

    def relocate_model(
        self, model_name: str, new_base_path: str, move_files: bool = False
    ) -> Dict[str, Any]:
        """Relocate an existing registered model to a custom directory (either re-pointing path or moving files on disk)."""
        new_base_path = os.path.abspath(new_base_path.strip().strip('"'))
        with self._lock:
            if model_name not in self.state["models"]:
                return {"ok": False, "error": f"未找到模型: {model_name}"}
            entry = self.state["models"][model_name]
            old_path = os.path.abspath(entry["base_path"])

        if _norm_path(old_path).lower() == _norm_path(new_base_path).lower():
            return {"ok": False, "error": "新路径与当前模型所在路径相同"}

        if move_files:
            if not os.path.isdir(old_path):
                return {"ok": False, "error": f"原模型目录不存在，无法迁移文件: {old_path}"}
            # Temporarily disable model in OVMS so file handles are released before moving
            was_enabled = bool(entry.get("enabled", True))
            if was_enabled:
                self.set_model_enabled(model_name, False)
                time.sleep(0.8)
            try:
                os.makedirs(os.path.dirname(new_base_path), exist_ok=True)
                if os.path.exists(new_base_path) and not os.listdir(new_base_path):
                    os.rmdir(new_base_path)
                shutil.move(old_path, new_base_path)
            except Exception as e:
                if was_enabled:
                    self.set_model_enabled(model_name, True)
                return {"ok": False, "error": f"移动模型文件失败 (可能仍被进程占用): {e}"}
        else:
            if not os.path.isdir(new_base_path):
                return {"ok": False, "error": f"指定的目标目录不存在: {new_base_path}"}

        info = inspect_model_directory(new_base_path)
        if not info.get("valid"):
            return {
                "ok": False,
                "error": f"目录 '{new_base_path}' 中未检测到有效的 OpenVINO IR (.xml/.bin) 或 .gguf 模型文件",
            }

        pbtxt_path = os.path.join(new_base_path, "graph.pbtxt")
        if not os.path.exists(pbtxt_path):
            Path(pbtxt_path).write_text(build_graph_pbtxt(entry.get("params") or {}), encoding="utf-8")
            info = inspect_model_directory(new_base_path)

        with self._lock:
            if not move_files:
                # Exclude old path so auto-scan doesn't re-add the old location as a duplicate
                excl = self.state.setdefault("excluded_paths", [])
                if _norm_path(old_path) not in excl:
                    excl.append(_norm_path(old_path))
            self.state["models"][model_name].update({
                "base_path": info["base_path"],
                "size_gb": info["size_gb"],
                "task_type": info["task_type"],
                "architecture": info["architecture"],
                "quantization": info["quantization"],
                "has_pbtxt": True,
                "params": info["params"],
            })
            if move_files:
                self.state["models"][model_name]["enabled"] = was_enabled
            self._save_state()
            self._write_ovms_config_json()

        reload_res = self.trigger_ovms_config_reload()
        return {
            "ok": True,
            "model_name": model_name,
            "old_path": _norm_path(old_path),
            "new_path": info["base_path"],
            "moved_files": move_files,
            "reload": reload_res,
            "models": self.get_models_with_live_status(),
        }

    def remove_model(self, model_name: str, delete_files: bool = False) -> Dict[str, Any]:
        with self._lock:
            if model_name not in self.state["models"]:
                return {"ok": False, "error": f"未找到模型: {model_name}"}
            entry = self.state["models"].pop(model_name)
            norm_bp = _norm_path(entry.get("base_path", ""))
            if not delete_files and norm_bp:
                excl = self.state.setdefault("excluded_paths", [])
                if norm_bp not in excl:
                    excl.append(norm_bp)
            self._save_state()
            self._write_ovms_config_json()

        self.trigger_ovms_config_reload()
        deleted_disk = False
        delete_error = ""
        target_dir = os.path.abspath(entry["base_path"]) if entry.get("base_path") else ""
        if delete_files and target_dir and os.path.isdir(target_dir):
            # Wait briefly for OVMS config reload to unload model handles
            for attempt in range(4):
                time.sleep(0.5)
                try:
                    shutil.rmtree(target_dir)
                    deleted_disk = True
                    break
                except Exception as e:
                    delete_error = str(e)
            if not deleted_disk and os.path.isdir(target_dir):
                shutil.rmtree(target_dir, ignore_errors=True)
                deleted_disk = not os.path.exists(target_dir)

        return {
            "ok": True,
            "removed": model_name,
            "deleted_files": deleted_disk,
            "delete_error": delete_error if (delete_files and not deleted_disk) else "",
            "models": self.get_models_with_live_status(),
        }

    def update_settings(self, new_settings: Dict[str, Any]) -> Dict[str, Any]:
        rescan_needed = False
        restart_if_running = bool(new_settings.pop("restart_if_running", False))
        endpoint_changed = False

        # Allow combined rest_api_address like "127.0.0.1:8000", "0.0.0.0:8080", or "http://192.168.1.50:8000/v3"
        raw_api_addr = new_settings.pop("rest_api_address", None)
        if raw_api_addr and isinstance(raw_api_addr, str) and raw_api_addr.strip():
            s = raw_api_addr.strip()
            if "://" not in s:
                s = "http://" + s
            parsed = urllib.parse.urlparse(s)
            if parsed.hostname:
                new_settings.setdefault("rest_bind_address", parsed.hostname)
            if parsed.port:
                new_settings.setdefault("rest_port", int(parsed.port))

        with self._lock:
            old_rest_bind = str(self.state["settings"].get("rest_bind_address") or "127.0.0.1")
            old_rest_port = int(self.state["settings"].get("rest_port") or 8000)
            old_grpc_bind = str(self.state["settings"].get("grpc_bind_address") or "127.0.0.1")
            old_grpc_port = int(self.state["settings"].get("grpc_port") or 9000)

            if "models_root" in new_settings and new_settings["models_root"]:
                raw_root = str(new_settings["models_root"]).strip().strip('"')
                abs_root = os.path.abspath(raw_root)
                os.makedirs(abs_root, exist_ok=True)
                new_settings["models_root"] = abs_root
                rescan_needed = True

            if "rest_bind_address" in new_settings and new_settings["rest_bind_address"] is not None:
                rb = str(new_settings["rest_bind_address"]).strip()
                if "://" in rb:
                    rb = urllib.parse.urlparse(rb).hostname or "127.0.0.1"
                elif ":" in rb and not rb.startswith("["):
                    rb = rb.split(":")[0].strip()
                if rb.lower() == "localhost":
                    rb = "127.0.0.1"
                new_settings["rest_bind_address"] = rb or "127.0.0.1"
                if "grpc_bind_address" not in new_settings:
                    new_settings["grpc_bind_address"] = new_settings["rest_bind_address"]

            if "grpc_bind_address" in new_settings and new_settings["grpc_bind_address"] is not None:
                gb = str(new_settings["grpc_bind_address"]).strip()
                if gb.lower() == "localhost":
                    gb = "127.0.0.1"
                new_settings["grpc_bind_address"] = gb or "127.0.0.1"

            if "rest_port" in new_settings and new_settings["rest_port"] is not None:
                try:
                    rp = int(new_settings["rest_port"])
                except (ValueError, TypeError):
                    return {"ok": False, "error": "REST 端口号必须为整数 (1 ~ 65535)"}
                if rp < 1 or rp > 65535:
                    return {"ok": False, "error": "REST 端口号超出有效范围 (1 ~ 65535)"}
                new_settings["rest_port"] = rp

            if "grpc_port" in new_settings and new_settings["grpc_port"] is not None:
                try:
                    gp = int(new_settings["grpc_port"])
                except (ValueError, TypeError):
                    return {"ok": False, "error": "gRPC 端口号必须为整数 (1 ~ 65535)"}
                if gp < 1 or gp > 65535:
                    return {"ok": False, "error": "gRPC 端口号超出有效范围 (1 ~ 65535)"}
                new_settings["grpc_port"] = gp

            final_rp = int(new_settings.get("rest_port", old_rest_port))
            final_gp = int(new_settings.get("grpc_port", old_grpc_port))
            if final_rp == final_gp:
                if "grpc_port" not in new_settings:
                    new_settings["grpc_port"] = final_rp + 1000 if final_rp <= 64535 else final_rp - 1
                else:
                    return {"ok": False, "error": "REST 端口与 gRPC 端口不能相同"}

            if "shared_vram_gb" in new_settings and new_settings["shared_vram_gb"] is not None:
                try:
                    val = round(max(1.0, min(128.0, float(new_settings["shared_vram_gb"]))), 2)
                except (ValueError, TypeError):
                    val = 16.0
                new_settings["shared_vram_gb"] = val
                if "ram_budget_gb" not in new_settings:
                    new_settings["ram_budget_gb"] = val
            if "ram_budget_gb" in new_settings and new_settings["ram_budget_gb"] is not None:
                try:
                    rval = round(max(1.0, min(128.0, float(new_settings["ram_budget_gb"]))), 2)
                except (ValueError, TypeError):
                    rval = 16.0
                new_settings["ram_budget_gb"] = rval
                if "shared_vram_gb" not in new_settings:
                    new_settings["shared_vram_gb"] = rval

            self.state["settings"].update(new_settings)
            self._save_state()

            endpoint_changed = (
                self.state["settings"].get("rest_bind_address") != old_rest_bind
                or int(self.state["settings"].get("rest_port", 8000)) != old_rest_port
                or self.state["settings"].get("grpc_bind_address") != old_grpc_bind
                or int(self.state["settings"].get("grpc_port", 9000)) != old_grpc_port
            )

        if rescan_needed:
            self.discover_and_sync_models(write_ovms_config=True)

        restarted = False
        if restart_if_running and endpoint_changed:
            st = self.get_server_status()
            if st["running"]:
                self.restart_server()
                restarted = True

        return {
            "ok": True,
            "restarted": restarted,
            "endpoint_changed": endpoint_changed,
            "settings": self.state["settings"],
            "server": self.get_server_status(),
            "models": self.get_models_with_live_status(),
        }

    def clear_shader_cache(self) -> Dict[str, Any]:
        cache_dir = self.state["settings"].get("cache_dir", DEFAULT_CACHE_DIR)
        removed_mb = 0.0
        if os.path.isdir(cache_dir):
            for r, _, fs in os.walk(cache_dir):
                for f in fs:
                    fp = os.path.join(r, f)
                    try:
                        removed_mb += os.path.getsize(fp) / (1024 * 1024)
                        os.remove(fp)
                    except OSError:
                        pass
        return {"ok": True, "cleared_mb": round(removed_mb, 1)}

    def start_model_download(
        self,
        repo_id: str,
        source: str = "modelscope",
        target_device: str = "GPU",
        auto_enable: bool = True,
        dest_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        repo_id = repo_id.strip().strip("/")
        if not repo_id or "/" not in repo_id:
            return {"ok": False, "error": "请输入有效的仓库 ID，例如 OpenVINO/Qwen3-4B-int4-ov"}

        model_name = repo_id.split("/")[-1]
        base_root = os.path.abspath(dest_root.strip().strip('"')) if (dest_root and dest_root.strip()) else self.state["settings"]["models_root"]
        dest_dir = os.path.join(base_root, model_name)
        job_id = f"dl_{model_name}_{int(time.time())}"

        job_info = {
            "job_id": job_id,
            "repo_id": repo_id,
            "model_name": model_name,
            "source": source,
            "dest_dir": _norm_path(dest_dir),
            "status": "starting",
            "progress_pct": 0.0,
            "downloaded_mb": 0.0,
            "total_mb": 0.0,
            "speed_mbps": 0.0,
            "current_file": "正在获取仓库文件清单...",
            "error": "",
            "cancel_flag": False,
            "started_at": time.time(),
        }
        self.downloads[job_id] = job_info

        t = threading.Thread(
            target=self._download_worker,
            args=(job_id, repo_id, source, dest_dir, target_device, auto_enable),
            daemon=True,
        )
        t.start()
        return {"ok": True, "job": job_info}

    def cancel_download(self, job_id: str) -> Dict[str, Any]:
        if job_id in self.downloads:
            self.downloads[job_id]["cancel_flag"] = True
            self.downloads[job_id]["status"] = "cancelled"
            return {"ok": True}
        return {"ok": False, "error": "任务不存在"}

    def _download_worker(
        self, job_id: str, repo_id: str, source: str, dest_dir: str, target_device: str, auto_enable: bool
    ):
        job = self.downloads[job_id]
        try:
            os.makedirs(dest_dir, exist_ok=True)
            files_to_get = []

            if source == "modelscope":
                api_url = f"https://www.modelscope.cn/api/v1/models/{repo_id}/repo/files"
                req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15) as r:
                    data = json.loads(r.read().decode("utf-8"))
                raw_files = data.get("Data", {}).get("Files", [])
                for rf in raw_files:
                    if rf.get("Type") == "blob" and not rf.get("Path", "").startswith(".git"):
                        rel_path = rf["Path"]
                        dl_url = f"https://www.modelscope.cn/api/v1/models/{repo_id}/repo?Revision=master&FilePath={urllib.parse.quote(rel_path)}"
                        files_to_get.append((rel_path, dl_url, int(rf.get("Size") or 0)))
            else:
                base_host = "https://hf-mirror.com" if source == "hf-mirror" else "https://huggingface.co"
                api_url = f"{base_host}/api/models/{repo_id}/tree/main"
                req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15) as r:
                    raw_files = json.loads(r.read().decode("utf-8"))
                for rf in raw_files:
                    if rf.get("type") == "file" and not rf.get("path", "").startswith(".git"):
                        rel_path = rf["path"]
                        dl_url = f"{base_host}/{repo_id}/resolve/main/{urllib.parse.quote(rel_path)}"
                        files_to_get.append((rel_path, dl_url, int(rf.get("size") or 0)))

            if not files_to_get:
                raise RuntimeError(f"未在 {source} 找到模型仓库文件: {repo_id}")

            total_bytes = sum(x[2] for x in files_to_get)
            job["total_mb"] = round(total_bytes / (1024 * 1024), 1)
            job["status"] = "downloading"

            downloaded_bytes = 0
            start_t = time.time()

            for rel_path, dl_url, fsize in files_to_get:
                if job.get("cancel_flag"):
                    job["status"] = "cancelled"
                    return

                job["current_file"] = rel_path
                target_file = os.path.join(dest_dir, rel_path)
                os.makedirs(os.path.dirname(target_file), exist_ok=True)

                # Resume/skip if already downloaded with matching size
                if os.path.exists(target_file) and fsize > 0 and os.path.getsize(target_file) == fsize:
                    downloaded_bytes += fsize
                    job["downloaded_mb"] = round(downloaded_bytes / (1024 * 1024), 1)
                    job["progress_pct"] = round(downloaded_bytes * 100.0 / max(1, total_bytes), 1)
                    continue

                req = urllib.request.Request(dl_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp, open(target_file + ".part", "wb") as out_f:
                    while True:
                        if job.get("cancel_flag"):
                            job["status"] = "cancelled"
                            return
                        chunk = resp.read(1024 * 512)
                        if not chunk:
                            break
                        out_f.write(chunk)
                        downloaded_bytes += len(chunk)
                        elapsed = max(0.1, time.time() - start_t)
                        job["downloaded_mb"] = round(downloaded_bytes / (1024 * 1024), 1)
                        job["speed_mbps"] = round((downloaded_bytes / (1024 * 1024)) / elapsed, 2)
                        if total_bytes > 0:
                            job["progress_pct"] = min(99.9, round(downloaded_bytes * 100.0 / total_bytes, 1))

                os.replace(target_file + ".part", target_file)

            # Create graph.pbtxt and register
            pbtxt_path = os.path.join(dest_dir, "graph.pbtxt")
            if not os.path.exists(pbtxt_path):
                Path(pbtxt_path).write_text(
                    build_graph_pbtxt({"device": target_device, "kv_cache_precision": "f16"}),
                    encoding="utf-8",
                )

            self.import_local_model(dest_dir, custom_name=job["model_name"], enable=auto_enable, default_device=target_device)
            job["progress_pct"] = 100.0
            job["current_file"] = "下载完成并已自动配置 graph.pbtxt"
            job["status"] = "completed"
        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)

    def get_logs(self, lines: int = 120) -> Dict[str, Any]:
        def _tail(fp: str, n: int) -> str:
            if not os.path.exists(fp):
                return ""
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    all_lines = f.readlines()
                    return "".join(all_lines[-n:])
            except Exception as e:
                return str(e)

        stdout_txt = _tail(STDOUT_LOG, lines)
        stderr_txt = _tail(STDERR_LOG, 60)

        # Extract latest KV cache usage line if present
        kv_matches = re.findall(r"Cache type:\s*([^,]+),\s*cache usage:\s*([\d.]+)%\s*of\s*([\d.]+\s*\w+)", stdout_txt)
        kv_summary = None
        if kv_matches:
            ctype, pct, cap = kv_matches[-1]
            kv_summary = {"cache_type": ctype, "usage_pct": float(pct), "capacity": cap}

        return {
            "stdout": stdout_txt,
            "stderr": stderr_txt,
            "kv_cache": kv_summary,
        }

    def _initial_update_check(self):
        time.sleep(2.0)
        try:
            res = self.check_ovms_update(force=True)
            if res.get("update_available") and self.state["settings"].get("auto_apply_updates", False):
                self.start_ovms_update()
        except Exception:
            pass

    @staticmethod
    def _parse_ovms_version_text(raw_text: str) -> Dict[str, str]:
        info = {
            "ovms_version": "",
            "ovms_commit": "",
            "openvino_backend": "",
            "genai_backend": "",
            "raw": raw_text.strip(),
        }
        for line in raw_text.splitlines():
            s = line.strip()
            if s.startswith("OpenVINO Model Server"):
                ver = s.replace("OpenVINO Model Server", "").strip()
                info["ovms_version"] = ver
                parts = ver.split(".")
                if len(parts) >= 4:
                    info["ovms_commit"] = parts[-1]
            elif s.startswith("OpenVINO backend"):
                info["openvino_backend"] = s.replace("OpenVINO backend", "").strip()
            elif s.startswith("OpenVINO GenAI backend"):
                info["genai_backend"] = s.replace("OpenVINO GenAI backend", "").strip()
        return info

    @staticmethod
    def _extract_numeric_tuple(ver_str: str):
        nums = re.findall(r"\d+", ver_str or "")
        return tuple(int(x) for x in nums[:3]) if nums else (0, 0, 0)

    def get_installed_ovms_version(self, force: bool = False) -> Dict[str, str]:
        if self._installed_version_cache and not force:
            return self._installed_version_cache
        ovms_dir = self.state["settings"].get("ovms_dir", DEFAULT_OVMS_DIR)
        ovms_exe = os.path.join(ovms_dir, "ovms.exe")
        if not os.path.exists(ovms_exe):
            res = {
                "ovms_version": "未安装",
                "ovms_commit": "",
                "openvino_backend": "",
                "genai_backend": "",
                "raw": "",
            }
            self._installed_version_cache = res
            return res
        try:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) if os.name == "nt" else 0
            out = subprocess.check_output(
                [ovms_exe, "--version"],
                cwd=ovms_dir,
                text=True,
                stderr=subprocess.STDOUT,
                timeout=6,
                creationflags=creationflags,
            )
            parsed = self._parse_ovms_version_text(out)
            self._installed_version_cache = parsed
            return parsed
        except Exception as e:
            res = {
                "ovms_version": "读取失败",
                "ovms_commit": "",
                "openvino_backend": "",
                "genai_backend": "",
                "raw": str(e),
            }
            self._installed_version_cache = res
            return res

    def get_updater_status(self) -> Dict[str, Any]:
        installed = self.get_installed_ovms_version(force=False)
        self.updater_state["installed"] = installed
        return {
            **self.updater_state,
            "channel": self.state["settings"].get("update_channel", "weekly"),
            "auto_check_updates": bool(self.state["settings"].get("auto_check_updates", True)),
            "check_interval_hours": float(self.state["settings"].get("check_interval_hours", 6)),
            "auto_apply_updates": bool(self.state["settings"].get("auto_apply_updates", False)),
        }

    def check_ovms_update(self, channel: Optional[str] = None, force: bool = True) -> Dict[str, Any]:
        if self.updater_state.get("checking") or self.updater_state.get("updating"):
            return {"ok": True, **self.get_updater_status()}

        use_channel = channel or self.state["settings"].get("update_channel", "weekly")
        self.updater_state["checking"] = True
        self.updater_state["error"] = ""
        try:
            installed = self.get_installed_ovms_version(force=force)
            self.updater_state["installed"] = installed
            local_ver = installed.get("ovms_version", "")

            remote_info: Dict[str, Any] = {"channel": use_channel}
            update_available = False

            if use_channel == "weekly":
                req_v = urllib.request.Request(WEEKLY_VERSION_URL, headers={"User-Agent": "OpenVINO-Studio/2.0"})
                with urllib.request.urlopen(req_v, timeout=10) as r:
                    ver_txt = r.read().decode("utf-8", errors="ignore")
                parsed_remote = self._parse_ovms_version_text(ver_txt)

                sha256_hex = ""
                try:
                    req_s = urllib.request.Request(WEEKLY_SHA256_URL, headers={"User-Agent": "OpenVINO-Studio/2.0"})
                    with urllib.request.urlopen(req_s, timeout=8) as r:
                        sha256_hex = r.read().decode("utf-8", errors="ignore").strip().split()[0]
                except Exception:
                    pass

                last_mod = ""
                size_mb = 0.0
                try:
                    req_h = urllib.request.Request(
                        WEEKLY_ZIP_URL, headers={"User-Agent": "OpenVINO-Studio/2.0"}, method="HEAD"
                    )
                    with urllib.request.urlopen(req_h, timeout=8) as r:
                        last_mod = r.headers.get("Last-Modified", "")
                        clen = int(r.headers.get("Content-Length") or 0)
                        if clen > 0:
                            size_mb = round(clen / (1024 * 1024), 1)
                except Exception:
                    pass

                remote_ver = parsed_remote.get("ovms_version", "")
                remote_info.update({
                    "ovms_version": remote_ver,
                    "openvino_backend": parsed_remote.get("openvino_backend", ""),
                    "genai_backend": parsed_remote.get("genai_backend", ""),
                    "download_url": WEEKLY_ZIP_URL,
                    "sha256": sha256_hex,
                    "published_at": last_mod,
                    "size_mb": size_mb,
                })
                if remote_ver and local_ver in ("", "未安装", "读取失败"):
                    update_available = True
                elif remote_ver and remote_ver != local_ver:
                    # Compare tuple first; if same major.minor.patch, different commit hash means newer weekly build
                    l_tup = self._extract_numeric_tuple(local_ver)
                    r_tup = self._extract_numeric_tuple(remote_ver)
                    if r_tup >= l_tup:
                        update_available = True
            else:
                req_g = urllib.request.Request(GITHUB_LATEST_RELEASE_API, headers={"User-Agent": "OpenVINO-Studio/2.0"})
                with urllib.request.urlopen(req_g, timeout=12) as r:
                    rel = json.loads(r.read().decode("utf-8", errors="ignore"))
                tag = (rel.get("tag_name") or "").lstrip("v")
                assets = rel.get("assets") or []
                chosen_asset = None
                sha_asset = None
                # Prefer python_on package to match Cherry Studio / full MediaPipe capabilities
                for a in assets:
                    nm = (a.get("name") or "").lower()
                    if nm.startswith("ovms_windows") and nm.endswith(".zip") and "python_on" in nm:
                        chosen_asset = a
                    elif nm.startswith("ovms_windows") and nm.endswith(".zip.sha256") and "python_on" in nm:
                        sha_asset = a
                if not chosen_asset:
                    for a in assets:
                        nm = (a.get("name") or "").lower()
                        if nm.startswith("ovms_windows") and nm.endswith(".zip"):
                            chosen_asset = a
                            break

                sha256_hex = ""
                if sha_asset and sha_asset.get("browser_download_url"):
                    try:
                        req_s = urllib.request.Request(
                            sha_asset["browser_download_url"], headers={"User-Agent": "OpenVINO-Studio/2.0"}
                        )
                        with urllib.request.urlopen(req_s, timeout=8) as r:
                            sha256_hex = r.read().decode("utf-8", errors="ignore").strip().split()[0]
                    except Exception:
                        pass

                dl_url = chosen_asset["browser_download_url"] if chosen_asset else ""
                size_mb = round(int(chosen_asset.get("size") or 0) / (1024 * 1024), 1) if chosen_asset else 0.0
                remote_info.update({
                    "ovms_version": tag,
                    "openvino_backend": f"Release {tag}",
                    "genai_backend": f"Release {tag}",
                    "download_url": dl_url,
                    "sha256": sha256_hex,
                    "published_at": rel.get("published_at", ""),
                    "size_mb": size_mb,
                    "release_url": rel.get("html_url", ""),
                })
                l_tup = self._extract_numeric_tuple(local_ver)
                r_tup = self._extract_numeric_tuple(tag)
                if local_ver in ("", "未安装", "读取失败") and tag:
                    update_available = True
                elif r_tup > l_tup:
                    update_available = True

            self.updater_state["remote"] = remote_info
            self.updater_state["update_available"] = update_available
            self.updater_state["last_checked_ts"] = time.time()
            self.updater_state["message"] = (
                f"发现新版本 {remote_info.get('ovms_version')}"
                if update_available
                else f"当前已是最新版本 ({local_ver})"
            )
            return {"ok": True, **self.get_updater_status()}
        except Exception as e:
            self.updater_state["error"] = f"检查更新失败: {e}"
            return {"ok": False, "error": self.updater_state["error"], **self.get_updater_status()}
        finally:
            self.updater_state["checking"] = False

    def start_ovms_update(self, channel: Optional[str] = None, force_reinstall: bool = False) -> Dict[str, Any]:
        if self.updater_state.get("updating"):
            return {"ok": False, "error": "已有正在进行的 OpenVINO 更新任务", **self.get_updater_status()}

        use_channel = channel or self.state["settings"].get("update_channel", "weekly")
        # Ensure remote metadata is fresh
        chk = self.check_ovms_update(channel=use_channel, force=True)
        if not chk.get("ok"):
            return chk

        if not self.updater_state.get("update_available") and not force_reinstall:
            return {
                "ok": True,
                "already_latest": True,
                "message": "当前安装的 OpenVINO Model Server 已是最新版本",
                **self.get_updater_status(),
            }

        dl_url = (self.updater_state.get("remote") or {}).get("download_url")
        if not dl_url:
            return {"ok": False, "error": "未获取到有效的官方 Windows 安装包下载地址"}

        self.updater_state.update({
            "updating": True,
            "phase": "downloading",
            "progress_pct": 0.0,
            "downloaded_mb": 0.0,
            "total_mb": (self.updater_state.get("remote") or {}).get("size_mb", 0.0),
            "speed_mbps": 0.0,
            "error": "",
            "message": "正在连接官方源下载 OpenVINO Model Server 更新包...",
        })
        t = threading.Thread(target=self._ovms_update_worker, args=(use_channel,), daemon=True)
        t.start()
        return {"ok": True, "started": True, **self.get_updater_status()}

    def _ovms_update_worker(self, channel: str):
        was_running = False
        ovms_dir = self.state["settings"].get("ovms_dir", DEFAULT_OVMS_DIR)
        cfg_path = self.state["settings"].get("config_json_path", DEFAULT_CONFIG_JSON)
        tmp_zip = os.path.join(STUDIO_DIR, f"ovms_update_{channel}.zip.part")
        final_zip = os.path.join(STUDIO_DIR, f"ovms_update_{channel}.zip")
        extract_dir = os.path.join(STUDIO_DIR, "ovms_update_staging")

        try:
            remote = self.updater_state.get("remote") or {}
            dl_url = remote.get("download_url", "")
            expected_sha = (remote.get("sha256") or "").lower().strip()

            req = urllib.request.Request(dl_url, headers={"User-Agent": "OpenVINO-Studio/2.0"})
            hasher = hashlib.sha256()
            downloaded = 0
            start_t = time.time()

            with urllib.request.urlopen(req, timeout=45) as resp, open(tmp_zip, "wb") as out_f:
                total_bytes = int(resp.headers.get("Content-Length") or 0)
                if total_bytes > 0:
                    self.updater_state["total_mb"] = round(total_bytes / (1024 * 1024), 1)
                while True:
                    chunk = resp.read(1024 * 512)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    hasher.update(chunk)
                    downloaded += len(chunk)
                    elapsed = max(0.1, time.time() - start_t)
                    self.updater_state["downloaded_mb"] = round(downloaded / (1024 * 1024), 1)
                    self.updater_state["speed_mbps"] = round((downloaded / (1024 * 1024)) / elapsed, 2)
                    if total_bytes > 0:
                        self.updater_state["progress_pct"] = min(80.0, round(downloaded * 80.0 / total_bytes, 1))
                    self.updater_state["message"] = (
                        f"正在下载更新包: {self.updater_state['downloaded_mb']} / {self.updater_state['total_mb']} MB "
                        f"({self.updater_state['speed_mbps']} MB/s)"
                    )

            os.replace(tmp_zip, final_zip)

            if expected_sha and len(expected_sha) == 64:
                self.updater_state["phase"] = "verifying"
                self.updater_state["progress_pct"] = 83.0
                self.updater_state["message"] = "正在校验 SHA256 完整性签名..."
                actual_sha = hasher.hexdigest().lower()
                if actual_sha != expected_sha:
                    raise RuntimeError(f"安装包 SHA256 校验不匹配 (期望 {expected_sha[:12]}..., 实际 {actual_sha[:12]}...)")

            self.updater_state["phase"] = "extracting"
            self.updater_state["progress_pct"] = 87.0
            self.updater_state["message"] = "正在解压新版 OpenVINO 运行时组件..."
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir, ignore_errors=True)
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(final_zip, "r") as zf:
                zf.extractall(extract_dir)

            src_root = os.path.join(extract_dir, "ovms")
            if not os.path.exists(os.path.join(src_root, "ovms.exe")):
                # Fallback if zip extracts directly without 'ovms/' wrapper
                if os.path.exists(os.path.join(extract_dir, "ovms.exe")):
                    src_root = extract_dir
                else:
                    raise RuntimeError("解压后的目录中未找到 ovms.exe")

            # Backup existing config.json content in memory
            saved_cfg_bytes = None
            if os.path.exists(cfg_path):
                saved_cfg_bytes = Path(cfg_path).read_bytes()

            st = self.get_server_status()
            was_running = bool(st.get("running"))
            if was_running:
                self.updater_state["phase"] = "stopping"
                self.updater_state["progress_pct"] = 92.0
                self.updater_state["message"] = "正在安全停止运行中的 OVMS 服务以释放 DLL 文件锁..."
                self.stop_server()
                time.sleep(1.2)

            self.updater_state["phase"] = "installing"
            self.updater_state["progress_pct"] = 96.0
            self.updater_state["message"] = "正在覆盖升级 OpenVINO 二进制与 GenAI 插件..."
            os.makedirs(ovms_dir, exist_ok=True)
            shutil.copytree(src_root, ovms_dir, dirs_exist_ok=True)

            # Restore config.json so models remain untouched
            if saved_cfg_bytes is not None:
                os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
                Path(cfg_path).write_bytes(saved_cfg_bytes)
            else:
                self._write_ovms_config_json()

            # Clean staging
            shutil.rmtree(extract_dir, ignore_errors=True)
            try:
                os.remove(final_zip)
            except OSError:
                pass

            new_installed = self.get_installed_ovms_version(force=True)
            self.updater_state["installed"] = new_installed
            self.updater_state["update_available"] = False
            self.updater_state["last_updated_ts"] = time.time()

            if was_running:
                self.updater_state["phase"] = "restarting"
                self.updater_state["progress_pct"] = 99.0
                self.updater_state["message"] = "升级完成，正在自动恢复启动 OVMS 推理服务..."
                self.start_server()

            self.updater_state["phase"] = "completed"
            self.updater_state["progress_pct"] = 100.0
            self.updater_state["message"] = f"✔ 已成功升级至 OpenVINO {new_installed.get('ovms_version')}"
        except Exception as e:
            self.updater_state["phase"] = "failed"
            self.updater_state["error"] = str(e)
            self.updater_state["message"] = f"更新失败: {e}"
            if os.path.exists(tmp_zip):
                try:
                    os.remove(tmp_zip)
                except OSError:
                    pass
            if was_running and not self.get_server_status().get("running"):
                try:
                    self.start_server()
                except Exception:
                    pass
        finally:
            self.updater_state["updating"] = False

    def _watchdog_loop(self):
        while True:
            time.sleep(10.0)
            try:
                # 1. Periodic OVMS version auto-check & optional auto-update
                if self.state["settings"].get("auto_check_updates", True):
                    interval_h = max(0.5, float(self.state["settings"].get("check_interval_hours") or 6.0))
                    last_chk = float(self.updater_state.get("last_checked_ts") or 0.0)
                    if (time.time() - last_chk) >= (interval_h * 3600.0):
                        chk = self.check_ovms_update(force=True)
                        if chk.get("update_available") and self.state["settings"].get("auto_apply_updates", False):
                            self.start_ovms_update()

                # 2. Idle auto-stop check
                idle_min = float(self.state["settings"].get("idle_auto_stop_minutes") or 0)
                if idle_min <= 0:
                    continue
                st = self.get_server_status()
                if not st["running"]:
                    continue
                # Check if OVMS log had recent activity
                if os.path.exists(STDOUT_LOG):
                    mtime = os.path.getmtime(STDOUT_LOG)
                    self.last_active_ts = max(self.last_active_ts, mtime)
                if (time.time() - self.last_active_ts) >= (idle_min * 60.0):
                    self.stop_server()
            except Exception:
                pass
