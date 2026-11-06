# ⚡ OpenVINO Studio

**轻量级 OpenVINO Model Server (OVMS) 桌面与 Web 可视化控制中心**

专为 Intel CPU / Iris Xe / Arc 核显与独显以及全平台 NPU（Intel AI Boost / AMD Ryzen AI / Qualcomm Hexagon）打造的本地大模型可视化管理面板。支持一键启停服务、模型热插拔开关、MediaPipe `graph.pbtxt` 参数可视化调优、在线高速拉取预量化模型、通用底层硬件算力遥测以及 OVMS 在线自动检测与无损升级，并与 **Cherry Studio** 等 OpenAI 兼容客户端 100% 无缝协同。

---

## ✨ 核心特性

- **🟢 服务一键启停与零驻留释放**
  - 一键启动、停止、重启 OpenVINO Model Server，停服即刻 100% 释放系统内存与核显共享显存。
  - 支持 **一键安全退出 OpenVINO Studio 控制台**（可选择“同时停止 OVMS 释放全部显存/内存”或“仅退出 Studio 界面并保持 OVMS 后台继续服务”）。
  - 支持 **自定义 REST API 监听地址与端口**（`127.0.0.1` 本机模式 / `0.0.0.0` 局域网共享模式 / 自定义端口与 gRPC 端口），保存后可自动热重启生效。
  - 支持 **Watchdog 空闲自动休眠省电保护**（15 / 30 / 60 分钟无请求自动停服释放内存）。
- **📦 模型热插拔与灵活路径管理**
  - **独立模型开关**：勾选即启用并通知 OVMS 热重载（`POST /v1/config/reload`），取消勾选即卸载释放内存，与 Cherry Studio `config.json` 实时同步。
  - **灵活存储位置**：支持自定义全局默认模型库根目录、单模型独立更改路径（含一键剪切迁移文件）、前端移除列表或彻底删除本地模型文件。
  - **一键导入与国内直连下载**：支持扫描本地 OpenVINO IR / GGUF 目录自动生成 `graph.pbtxt`，内置 **ModelScope（阿里魔搭）** 与 **HF-Mirror** 高速下载器及精选 INT4 模型商店（Qwen3、Gemma 4、Phi-4 等）。
- **⚙ 可视化推理参数调优 (`graph.pbtxt`)**
  - 图形化调节推理设备（`GPU` / `CPU` / `NPU` / `AUTO` / `HETERO:GPU,CPU`）、KV Cache 精度（`f16` / `u8`）、最大并发数（`max_num_seqs`）、显存上限（`cache_size`）、前缀缓存（`enable_prefix_caching`）、性能策略（`LATENCY` / `THROUGHPUT`）及原始 `graph.pbtxt` 高级编辑。
- **📊 通用底层硬件实时遥测（零子进程开销）**
  - 基于 Windows 原生 `dxcore.dll` (WDDM/MCDM) 与 `pdh.dll` 性能计数器直采，实时监控 **CPU、GPU（3D/Compute 引擎及 OVMS 进程级核显占用）、NPU 利用率、系统内存** 及 **60 秒算力曲线**。
  - 支持 **自定义共享显存大小 / 模型显存预算上限**（默认 `16 GB`，支持 `1 ~ 128 GB` 自由设定与快捷预设）。
- **🔄 OpenVINO 引擎版本自动检测与更新中心**
  - 支持 **Weekly 前沿构建版**（第一时间适配 Gemma 4 等最新算子）与 **GitHub Stable 正式版** 双通道在线检测。
  - 提供 **手动确认一键升级** 与 **全自动无感更新** 两种策略，内置官方 **SHA256 完整性校验** 与 `models/config.json` 配置无损保护。
- **🚀 在线对话与吐字测速台**
  - 内置流式推理测试面板，实时统计 **首字延迟 (TTFT)** 与 **生成速率 (tokens/s)**，并提供持久化 OpenCL/GPU 着色器编译缓存（`CACHE_DIR`）加速二次启动。

---

## 🛠 环境要求

- **操作系统**：Windows 10 / 11 (x64)
- **Python**：Python 3.10+
- **OpenVINO Model Server (OVMS)**：
  - 若已安装 Cherry Studio 内置 OVMS（默认路径 `%USERPROFILE%\.cherrystudio\ovms\ovms`），程序将自动识别并复用；
  - 也可通过环境变量 `OVMS_DIR` 指定任意本地 OVMS 解压目录，或直接在面板「🔄 版本更新」中一键在线下载安装最新版。

---

## 🚀 快速开始

### 1. 克隆仓库并安装依赖

```bash
git clone https://github.com/lilynas/openvino-studio.git
cd openvino-studio
pip install -r requirements.txt
```

### 2. 启动控制台

- **桌面独立窗口模式（推荐）**：直接双击运行 `start_studio.bat`，或执行：
  ```bash
  python launch_studio.py
  ```
  后台将静默启动 FastAPI 服务（默认 `http://127.0.0.1:18080`），并自动唤起无边框原生感桌面应用窗口。

- **纯后端 Web 服务模式**：
  ```bash
  python launch_studio.py --server-only
  ```
  随后在任意浏览器访问 `http://127.0.0.1:18080`。

---

## 🔌 客户端接入（OpenAI 兼容 API）

启动服务后，OVMS 默认监听 `http://127.0.0.1:8000`，可直接接入 Cherry Studio、Open WebUI、Cursor、Continue 等任意支持 OpenAI 协议的客户端：

- **API Base URL**：`http://127.0.0.1:8000/v3`
- **Chat Completions**：`http://127.0.0.1:8000/v3/chat/completions`
- **API Key**：任意非空字符串（本地免鉴权，如 `ovms`）

---

## ⚙ 可选环境变量

| 环境变量 | 说明 | 默认值 |
| --- | --- | --- |
| `OVMS_STUDIO_HOST` | Studio Web 控制台监听地址 | `127.0.0.1` |
| `OVMS_STUDIO_PORT` | Studio Web 控制台监听端口 | `18080` |
| `OVMS_DIR` | OpenVINO Model Server (`ovms.exe`) 所在目录 | `~/.cherrystudio/ovms/ovms` |
| `OVMS_MODELS_ROOT` | 默认模型扫描与下载存储根目录 | 自动识别上级 `LLMs` 目录或 `./models` |

---

## 📄 开源协议

本项目基于 [MIT License](LICENSE) 开源。
