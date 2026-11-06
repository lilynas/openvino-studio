import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ovms_manager import RECOMMENDED_CATALOG, OvmsManager, STUDIO_DIR
from telemetry import HardwareTelemetry

app = FastAPI(title="OpenVINO Studio Control Center", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

telemetry_engine = HardwareTelemetry(history_seconds=60)
ovms_mgr = OvmsManager()

STATIC_DIR = os.path.join(STUDIO_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ToggleModelReq(BaseModel):
    model_name: str
    enabled: bool


class UpdateParamsReq(BaseModel):
    params: Dict[str, Any] = {}
    raw_pbtxt: Optional[str] = None


class ImportModelReq(BaseModel):
    folder_path: str
    custom_name: Optional[str] = None
    enable: bool = True
    default_device: str = "GPU"


class StartDownloadReq(BaseModel):
    repo_id: str
    source: str = "modelscope"
    target_device: str = "GPU"
    auto_enable: bool = True
    dest_root: Optional[str] = None


class OpenFolderReq(BaseModel):
    path: Optional[str] = None
    model_name: Optional[str] = None


class RelocateModelReq(BaseModel):
    new_path: str
    move_files: bool = False


class BrowseFolderReq(BaseModel):
    initial_dir: Optional[str] = None
    title: Optional[str] = None


class OvmsUpdateCheckReq(BaseModel):
    channel: Optional[str] = None
    force: bool = True


class OvmsUpdateApplyReq(BaseModel):
    channel: Optional[str] = None
    force_reinstall: bool = False


class ChatBenchmarkReq(BaseModel):
    model: str
    prompt: str = "请用三句话简明介绍 Intel OpenVINO 的核心优势。"
    max_tokens: int = 200
    temperature: float = 0.7


@app.get("/", response_class=HTMLResponse)
def index_page():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h3>OpenVINO Studio UI is initializing...</h3>")


@app.get("/api/overview")
def get_overview():
    return {
        "server": ovms_mgr.get_server_status(),
        "models": ovms_mgr.get_models_with_live_status(),
        "telemetry": telemetry_engine.get_telemetry(),
        "downloads": list(ovms_mgr.downloads.values()),
        "catalog": RECOMMENDED_CATALOG,
    }


@app.post("/api/server/start")
def start_server():
    return ovms_mgr.start_server()


@app.post("/api/server/stop")
def stop_server():
    return ovms_mgr.stop_server()


@app.post("/api/server/restart")
def restart_server():
    return ovms_mgr.restart_server()


@app.post("/api/server/reload")
def reload_config():
    return ovms_mgr.trigger_ovms_config_reload()


@app.post("/api/models/scan")
def scan_models(clear_excluded: bool = False):
    return {
        "ok": True,
        "models": ovms_mgr.discover_and_sync_models(write_ovms_config=True, clear_excluded=clear_excluded),
    }


@app.post("/api/models/toggle")
def toggle_model(req: ToggleModelReq):
    return ovms_mgr.set_model_enabled(req.model_name, req.enabled)


@app.get("/api/models/{model_name}/pbtxt")
def get_model_pbtxt(model_name: str):
    return ovms_mgr.get_model_pbtxt(model_name)


@app.post("/api/models/{model_name}/params")
def update_model_params(model_name: str, req: UpdateParamsReq):
    return ovms_mgr.update_model_params(model_name, req.params, req.raw_pbtxt)


@app.post("/api/models/{model_name}/relocate")
def relocate_model(model_name: str, req: RelocateModelReq):
    return ovms_mgr.relocate_model(model_name, req.new_path, move_files=req.move_files)


@app.post("/api/models/import")
def import_model(req: ImportModelReq):
    return ovms_mgr.import_local_model(
        req.folder_path,
        custom_name=req.custom_name,
        enable=req.enable,
        default_device=req.default_device,
    )


@app.post("/api/models/browse")
def browse_local_folder(req: Optional[BrowseFolderReq] = None):
    """Open native Windows folder picker dialog."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        init_dir = (req.initial_dir if req and req.initial_dir else None) or ovms_mgr.state["settings"].get("models_root", r"D:\LLMs")
        if not os.path.isdir(init_dir):
            init_dir = r"D:\LLMs"
        dlg_title = (req.title if req and req.title else None) or "选择文件夹"

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(
            initialdir=init_dir,
            title=dlg_title,
        )
        root.destroy()
        return {"ok": True, "path": selected or ""}
    except Exception as e:
        return {"ok": False, "error": str(e), "path": ""}


@app.post("/api/models/open-folder")
def open_in_explorer(req: OpenFolderReq):
    raw_path = req.path
    if not raw_path and req.model_name:
        mentry = ovms_mgr.state["models"].get(req.model_name)
        if mentry:
            raw_path = mentry.get("base_path")
    if not raw_path:
        return {"ok": False, "error": "未提供有效路径或模型名称"}

    target = os.path.abspath(os.path.normpath(raw_path.strip().strip('"')))
    if os.path.exists(target):
        try:
            os.startfile(target)
        except Exception:
            subprocess.Popen(["explorer.exe", target])
        return {"ok": True, "path": target}
    return {"ok": False, "error": f"路径不存在: {target}"}


@app.delete("/api/models/{model_name}")
def remove_model(model_name: str, delete_files: bool = False):
    return ovms_mgr.remove_model(model_name, delete_files=delete_files)


@app.post("/api/downloads/start")
def start_download(req: StartDownloadReq):
    return ovms_mgr.start_model_download(
        repo_id=req.repo_id,
        source=req.source,
        target_device=req.target_device,
        auto_enable=req.auto_enable,
        dest_root=req.dest_root,
    )


@app.post("/api/downloads/{job_id}/cancel")
def cancel_download(job_id: str):
    return ovms_mgr.cancel_download(job_id)


@app.get("/api/logs")
def get_logs(lines: int = 150):
    return ovms_mgr.get_logs(lines=lines)


@app.post("/api/settings")
def update_settings(settings: Dict[str, Any]):
    return ovms_mgr.update_settings(settings)


@app.post("/api/cache/clear")
def clear_cache():
    return ovms_mgr.clear_shader_cache()


@app.get("/api/ovms/updater")
def get_updater_status():
    return {"ok": True, **ovms_mgr.get_updater_status()}


@app.post("/api/ovms/updater/check")
def check_ovms_update(req: OvmsUpdateCheckReq):
    return ovms_mgr.check_ovms_update(channel=req.channel, force=req.force)


@app.post("/api/ovms/updater/apply")
def apply_ovms_update(req: OvmsUpdateApplyReq):
    return ovms_mgr.start_ovms_update(channel=req.channel, force_reinstall=req.force_reinstall)


@app.post("/api/chat/benchmark")
def chat_benchmark(req: ChatBenchmarkReq):
    """
    Stream a test prompt to OVMS /v3/chat/completions and measure exact
    TTFT (Time-To-First-Token), total latency, token count, and tokens/sec.
    """
    port = ovms_mgr.state["settings"]["rest_port"]
    url = f"http://127.0.0.1:{port}/v3/chat/completions"
    payload = {
        "model": req.model,
        "messages": [{"role": "user", "content": req.prompt}],
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    body = json.dumps(payload).encode("utf-8")

    def event_stream():
        t0 = time.perf_counter()
        first_token_t = None
        chunks_count = 0
        completion_tokens = 0
        full_text = []

        try:
            http_req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(http_req, timeout=120) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        evt = json.loads(data_str)
                    except Exception:
                        continue

                    usage = evt.get("usage")
                    if usage and usage.get("completion_tokens"):
                        completion_tokens = int(usage["completion_tokens"])

                    choices = evt.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        piece = delta.get("content") or delta.get("reasoning_content") or ""
                        if piece:
                            now_t = time.perf_counter()
                            if first_token_t is None:
                                first_token_t = now_t
                            chunks_count += 1
                            full_text.append(piece)
                            ttft_ms = round((first_token_t - t0) * 1000.0, 1)
                            gen_elapsed = max(0.001, now_t - first_token_t)
                            est_toks = completion_tokens or chunks_count
                            tps = round(est_toks / gen_elapsed, 2) if chunks_count > 1 else 0.0
                            yield json.dumps(
                                {
                                    "type": "delta",
                                    "content": piece,
                                    "ttft_ms": ttft_ms,
                                    "tokens": est_toks,
                                    "tps": tps,
                                },
                                ensure_ascii=False,
                            ) + "\n"

            total_s = max(0.001, time.perf_counter() - t0)
            ttft_ms = round(((first_token_t or time.perf_counter()) - t0) * 1000.0, 1)
            final_toks = completion_tokens or chunks_count
            gen_window = max(0.001, (time.perf_counter() - first_token_t)) if first_token_t else total_s
            tps = round(final_toks / gen_window, 2)
            ovms_mgr.last_active_ts = time.time()
            yield json.dumps(
                {
                    "type": "done",
                    "ttft_ms": ttft_ms,
                    "total_sec": round(total_s, 2),
                    "tokens": final_toks,
                    "tps": tps,
                    "text": "".join(full_text),
                },
                ensure_ascii=False,
            ) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "error": str(e)}, ensure_ascii=False) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
