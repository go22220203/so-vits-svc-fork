from __future__ import annotations

import json
from logging import getLogger
from pathlib import Path

import librosa
import PySimpleGUI as sg
import sounddevice as sd
import torch
from pebble import ProcessFuture, ProcessPool
from tqdm.tk import tqdm_tk

from .__main__ import init_logger
from .utils import ensure_hubert_model

GUI_DEFAULT_PRESETS_PATH = Path(__file__).parent / "default_gui_presets.json"
GUI_PRESETS_PATH = Path("./user_gui_presets.json").absolute()
init_logger()

LOG = getLogger(__name__)

sg.set_options(font=('Arial', 12))

def play_audio(path: Path | str):
    if isinstance(path, Path):
        path = path.as_posix()
    data, sr = librosa.load(path)
    sd.play(data, sr)


def load_presets() -> dict:
    defaults = json.loads(GUI_DEFAULT_PRESETS_PATH.read_text())
    users = (
        json.loads(GUI_PRESETS_PATH.read_text()) if GUI_PRESETS_PATH.exists() else {}
    )
    # prioriy: defaults > users
    return {**defaults, **users}


def add_preset(name: str, preset: dict) -> dict:
    presets = load_presets()
    presets[name] = preset
    with GUI_PRESETS_PATH.open("w") as f:
        json.dump(presets, f, indent=2)
    return load_presets()


def delete_preset(name: str) -> dict:
    presets = load_presets()
    if name in presets:
        del presets[name]
    else:
        LOG.warning(f"Cannot delete preset {name} because it does not exist.")
    with GUI_PRESETS_PATH.open("w") as f:
        json.dump(presets, f, indent=2)
    return load_presets()


def get_devices(
    update: bool = True,
) -> tuple[list[str], list[str], list[int], list[int]]:
    if update:
        sd._terminate()
        sd._initialize()
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    for hostapi in hostapis:
        for device_idx in hostapi["devices"]:
            devices[device_idx]["hostapi_name"] = hostapi["name"]
    input_devices = [
        f"{d['name']} ({d['hostapi_name']})"
        for d in devices
        if d["max_input_channels"] > 0
    ]
    output_devices = [
        f"{d['name']} ({d['hostapi_name']})"
        for d in devices
        if d["max_output_channels"] > 0
    ]
    input_devices_indices = [d["index"] for d in devices if d["max_input_channels"] > 0]
    output_devices_indices = [
        d["index"] for d in devices if d["max_output_channels"] > 0
    ]
    return input_devices, output_devices, input_devices_indices, output_devices_indices


def main():
    try:
        ensure_hubert_model(tqdm_cls=tqdm_tk)
    except Exception as e:
        LOG.exception(e)
        LOG.info("Trying tqdm.std...")
        try:
            ensure_hubert_model()
        except Exception as e:
            LOG.exception(e)
            try:
                ensure_hubert_model(disable=True)
            except Exception as e:
                LOG.exception(e)
                LOG.error(
                    "Failed to download Hubert model. Please download it manually."
                )
                return

    sg.theme("Dark")
    model_candidates = list(sorted(Path("./logs/44k/").glob("G_*.pth")))

    frame_contents = {
        "Paths": [
            [
                sg.Text("模型文件"),
                sg.Push(),
                sg.InputText(
                    key="model_path",
                    default_text=model_candidates[-1].absolute().as_posix()
                    if model_candidates
                    else "",
                    enable_events=True,
                ),
                sg.FileBrowse(
                    initial_folder=Path("./logs/44k/").absolute
                    if Path("./logs/44k/").exists()
                    else Path(".").absolute().as_posix(),
                    key="model_path_browse",
                    file_types=(("PyTorch", "*.pth"),),
                ),
            ],
            [
                sg.Text("配置文件"),
                sg.Push(),
                sg.InputText(
                    key="config_path",
                    default_text=Path("./configs/44k/config.json").absolute().as_posix()
                    if Path("./configs/44k/config.json").exists()
                    else "",
                    enable_events=True,
                ),
                sg.FileBrowse(
                    initial_folder=Path("./configs/44k/").as_posix()
                    if Path("./configs/44k/").exists()
                    else Path(".").absolute().as_posix(),
                    key="config_path_browse",
                    file_types=(("JSON", "*.json"),),
                ),
            ],
            [
                sg.Text("聚类模型文件 (可选)"),
                sg.Push(),
                sg.InputText(
                    key="cluster_model_path",
                    default_text=Path("./logs/44k/kmeans.pt").absolute().as_posix()
                    if Path("./logs/44k/kmeans.pt").exists()
                    else "",
                    enable_events=True,
                ),
                sg.FileBrowse(
                    initial_folder="./logs/44k/"
                    if Path("./logs/44k/").exists()
                    else ".",
                    key="cluster_model_path_browse",
                    file_types=(("PyTorch", "*.pt"),),
                ),
            ],
        ],
        "Common": [
            [
                sg.Text("音色"),
                sg.Push(),
                sg.Combo(values=[], key="speaker", size=(20, 1)),
            ],
            [
                sg.Text("静音阈值"),
                sg.Push(),
                sg.Slider(
                    range=(-60.0, 0),
                    orientation="h",
                    key="silence_threshold",
                    resolution=0.1,
                ),
            ],
            [
                sg.Text(
                    "关闭自动预测基频功能的情况下\n"
                    "需要根据自己的声音情况来进行音高调整.",
                    size=(None, 3),
                ),
                sg.Push(),
                sg.Slider(
                    range=(-36, 36),
                    orientation="h",
                    key="transpose",
                    tick_interval=12,
                ),
            ],
            [
                sg.Checkbox(
                    key="auto_predict_f0",
                    text="自动f0预测，配合聚类模型f0预测效果更好,勾选会导致变调功能失效\n"
                    "（仅限转换语音，歌声不要勾选此项会究极跑调）",
                )
            ],
            [
                sg.Text("F0 基频预测方法"),
                sg.Push(),
                sg.Combo(
                    ["crepe", "crepe-tiny", "parselmouth", "dio", "harvest"],
                    key="f0_method",
                ),
            ],
            [
                sg.Text("聚类推理比率"),
                sg.Push(),
                sg.Slider(
                    range=(0, 1.0),
                    orientation="h",
                    key="cluster_infer_ratio",
                    resolution=0.01,
                ),
            ],
            [
                sg.Text("噪声比例"),
                sg.Push(),
                sg.Slider(
                    range=(0.0, 1.0),
                    orientation="h",
                    key="noise_scale",
                    resolution=0.01,
                ),
            ],
            [
                sg.Text("静音阈值"),
                sg.Push(),
                sg.Slider(
                    range=(0.0, 1.0),
                    orientation="h",
                    key="pad_seconds",
                    resolution=0.01,
                ),
            ],
            [
                sg.Text("切片秒数"),
                sg.Push(),
                sg.Slider(
                    range=(0.0, 3.0),
                    orientation="h",
                    key="chunk_seconds",
                    resolution=0.01,
                ),
            ],
            [
                sg.Checkbox(
                    key="absolute_thresh",
                    text="绝对阈值（要在实时推理中忽略（请勾选））",
                )
            ],
        ],
        "File": [
            [
                sg.Text("输入音频"),
                sg.Push(),
                sg.InputText(key="input_path"),
                sg.FileBrowse(initial_folder=".", key="input_path_browse"),
                sg.Button("播放", key="play_input"),
            ],
            [sg.Checkbox(key="auto_play", text="自动播放", default=True)],
        ],
        "Realtime": [
            [
                sg.Text("音频片段淡入淡出时间"),
                sg.Push(),
                sg.Slider(
                    range=(0, 0.6),
                    orientation="h",
                    key="crossfade_seconds",
                    resolution=0.001,
                ),
            ],
            [
                sg.Text(
                    "时域分割时间",  # \n(big -> more robust, slower, (the same) latency)"
                    tooltip="Big -> more robust, slower, (the same) latency",
                ),
                sg.Push(),
                sg.Slider(
                    range=(0, 3.0),
                    orientation="h",
                    key="block_seconds",
                    resolution=0.001,
                ),
            ],
            [
                sg.Text(
                    "推理延时（前）",  # \n(big -> more robust, slower)"
                    tooltip="Big -> more robust, slower, additional latency",
                ),
                sg.Push(),
                sg.Slider(
                    range=(0, 2.0),
                    orientation="h",
                    key="additional_infer_before_seconds",
                    resolution=0.001,
                ),
            ],
            [
                sg.Text(
                    "推理延时（后）",  # \n(big -> more robust, slower, additional latency)"
                    tooltip="Big -> more robust, slower, additional latency",
                ),
                sg.Push(),
                sg.Slider(
                    range=(0, 2.0),
                    orientation="h",
                    key="additional_infer_after_seconds",
                    resolution=0.001,
                ),
            ],
            [
                sg.Text("实时算法"),
                sg.Push(),
                sg.Combo(
                    ["按语音分段", "持续分段"],
                    default_value="持续分段",
                    key="realtime_algorithm",
                    size=(20, 1),
                ),
            ],
            [
                sg.Text("输入设备"),
                sg.Push(),
                sg.Combo(
                    key="input_device",
                    values=[],
                    size=(60, 1),
                ),
            ],
            [
                sg.Text("输出设备"),
                sg.Push(),
                sg.Combo(
                    key="output_device",
                    values=[],
                    size=(60, 1),
                ),
            ],
            [
                sg.Checkbox(
                    "直通原始音频（用于测试效果）",
                    key="passthrough_original",
                    default=False,
                ),
                sg.Push(),
                sg.Button("刷新设备", key="refresh_devices"),
            ],
            [
                sg.Frame(
                    "Notes",
                    [
                        [
                            sg.Text(
                                "在推理过程中:\n"
                                "    - 将 F0 预测方法设置为 'crepe' 可能会导致性能下降，效果变差\n"
                                "    - 必须关闭自动预测 F0\n"
                                "如果音频听起来含糊不清或杂乱无章:\n"
                                "    情况1：推理没有及时完成（增加时域分割时间）\n"
                                "    情况2：麦克风输入过低（降低静音阈值）\n"
                            )
                        ]
                    ],
                ),
            ],
        ],
        "Presets": [
            [
                sg.Text("预置参数"),
                sg.Push(),
                sg.Combo(
                    key="presets",
                    values=list(load_presets().keys()),
                    size=(20, 1),
                    enable_events=True,
                ),
                sg.Button("删除预设", key="delete_preset"),
            ],
            [
                sg.Text("预设名称"),
                sg.Stretch(),
                sg.InputText(key="preset_name", size=(20, 1)),
                sg.Button("将当前设置添加为预设", key="add_preset"),
            ],
        ],
    }

    # frames
    frames = {}
    for name, items in frame_contents.items():
        frame = sg.Frame(name, items)
        frame.expand_x = True
        frames[name] = [frame]

    column1 = sg.Column(
        [
            frames["Paths"],
            frames["Common"],
        ],
        vertical_alignment="top",
    )
    column2 = sg.Column(
        [
            frames["File"],
            frames["Realtime"],
            frames["Presets"],
            [
                sg.Checkbox(
                    key="use_gpu",
                    default=(
                        torch.cuda.is_available() or torch.backends.mps.is_available()
                    ),
                    text="使用GPU"
                    + (
                        " (not available; if your device has GPU, make sure you installed PyTorch with CUDA support)"
                        if not (
                            torch.cuda.is_available()
                            or torch.backends.mps.is_available()
                        )
                        else ""
                    ),
                    disabled=not (
                        torch.cuda.is_available() or torch.backends.mps.is_available()
                    ),
                )
            ],
            [
                sg.Button("推理", key="infer"),
                sg.Button("(重启)启动变声器", key="start_vc"),
                sg.Button("停止变声器", key="stop_vc"),
                sg.Push(),
                sg.Button("ONNX 导出", key="onnx_export"),
            ],
        ]
    )

    # columns
    layout = [[column1, column2]]
    # layout = [[sg.Column(layout, vertical_alignment="top", scrollable=True, expand_x=True, expand_y=True)]]
    window = sg.Window(
        f"{__name__.split('.')[0]}", layout, grab_anywhere=True, finalize=True
    )  # , use_custom_titlebar=True)
    # for n in ["input_device", "output_device"]:
    #     window[n].Widget.configure(justify="right")
    event, values = window.read(timeout=0.01)

    def update_speaker() -> None:
        from . import utils

        config_path = Path(values["config_path"])
        if config_path.exists() and config_path.is_file():
            hp = utils.get_hparams_from_file(values["config_path"])
            LOG.debug(f"Loaded config from {values['config_path']}")
            window["speaker"].update(
                values=list(hp.__dict__["spk"].keys()), set_to_index=0
            )

    def update_devices() -> None:
        input_devices, output_devices, _, _ = get_devices()
        window["input_device"].update(
            values=input_devices, value=values["input_device"]
        )
        window["output_device"].update(
            values=output_devices, value=values["output_device"]
        )
        input_default, output_default = sd.default.device
        if values["input_device"] not in input_devices:
            window["input_device"].update(
                values=input_devices,
                set_to_index=0 if input_default is None else input_default - 1,
            )
        if values["output_device"] not in output_devices:
            window["output_device"].update(
                values=output_devices,
                set_to_index=0 if output_default is None else output_default - 1,
            )

    PRESET_KEYS = [
        key
        for key in values.keys()
        if not any(exclude in key for exclude in ["preset", "browse"])
    ]

    def apply_preset(name: str) -> None:
        for key, value in load_presets()[name].items():
            if key in PRESET_KEYS:
                window[key].update(value)
                values[key] = value

    default_name = list(load_presets().keys())[0]
    apply_preset(default_name)
    window["presets"].update(default_name)
    del default_name
    update_speaker()
    update_devices()
    with ProcessPool(max_workers=1) as pool:
        future: None | ProcessFuture = None
        while True:
            event, values = window.read(200)
            if event == sg.WIN_CLOSED:
                break
            if not event == sg.EVENT_TIMEOUT:
                LOG.info(f"Event {event}, values {values}")
            if event.endswith("_path"):
                for name in window.AllKeysDict:
                    if str(name).endswith("_browse"):
                        browser = window[name]
                        if isinstance(browser, sg.Button):
                            LOG.info(
                                f"Updating browser {browser} to {Path(values[event]).parent}"
                            )
                            browser.InitialFolder = Path(values[event]).parent
                            browser.update()
                        else:
                            LOG.warning(f"Browser {browser} is not a FileBrowse")
            window["transpose"].update(
                disabled=values["auto_predict_f0"],
                visible=not values["auto_predict_f0"],
            )
            if event == "add_preset":
                presets = add_preset(
                    values["preset_name"], {key: values[key] for key in PRESET_KEYS}
                )
                window["presets"].update(values=list(presets.keys()))
            elif event == "delete_preset":
                presets = delete_preset(values["presets"])
                window["presets"].update(values=list(presets.keys()))
            elif event == "presets":
                apply_preset(values["presets"])
                update_speaker()
            elif event == "refresh_devices":
                update_devices()
            elif event == "config_path":
                update_speaker()
            elif event == "infer":
                from .inference_main import infer

                input_path = Path(values["input_path"])
                output_path = (
                    input_path.parent / f"{input_path.stem}.out{input_path.suffix}"
                )
                if not input_path.exists() or not input_path.is_file():
                    LOG.warning(f"Input path {input_path} does not exist.")
                    continue
                try:
                    infer(
                        model_path=Path(values["model_path"]),
                        config_path=Path(values["config_path"]),
                        input_path=input_path,
                        output_path=output_path,
                        speaker=values["speaker"],
                        cluster_model_path=Path(values["cluster_model_path"])
                        if values["cluster_model_path"]
                        else None,
                        transpose=values["transpose"],
                        auto_predict_f0=values["auto_predict_f0"],
                        cluster_infer_ratio=values["cluster_infer_ratio"],
                        noise_scale=values["noise_scale"],
                        db_thresh=values["silence_threshold"],
                        pad_seconds=values["pad_seconds"],
                        absolute_thresh=values["absolute_thresh"],
                        chunk_seconds=values["chunk_seconds"],
                        device="cpu"
                        if not values["use_gpu"]
                        else (
                            "cuda"
                            if torch.cuda.is_available()
                            else "mps"
                            if torch.backends.mps.is_available()
                            else "cpu"
                        ),
                    )
                    if values["auto_play"]:
                        pool.schedule(play_audio, args=[output_path])
                except Exception as e:
                    LOG.exception(e)
            elif event == "play_input":
                if Path(values["input_path"]).exists():
                    pool.schedule(play_audio, args=[Path(values["input_path"])])
            elif event == "start_vc":
                _, _, input_device_indices, output_device_indices = get_devices(
                    update=False
                )
                from .inference_main import realtime

                if future:
                    LOG.info("Canceling previous task")
                    future.cancel()
                future = pool.schedule(
                    realtime,
                    kwargs=dict(
                        model_path=Path(values["model_path"]),
                        config_path=Path(values["config_path"]),
                        speaker=values["speaker"],
                        cluster_model_path=Path(values["cluster_model_path"])
                        if values["cluster_model_path"]
                        else None,
                        transpose=values["transpose"],
                        auto_predict_f0=values["auto_predict_f0"],
                        cluster_infer_ratio=values["cluster_infer_ratio"],
                        noise_scale=values["noise_scale"],
                        f0_method=values["f0_method"],
                        crossfade_seconds=values["crossfade_seconds"],
                        additional_infer_before_seconds=values[
                            "additional_infer_before_seconds"
                        ],
                        additional_infer_after_seconds=values[
                            "additional_infer_after_seconds"
                        ],
                        db_thresh=values["silence_threshold"],
                        pad_seconds=values["pad_seconds"],
                        chunk_seconds=values["chunk_seconds"],
                        version=int(values["realtime_algorithm"][0]),
                        device="cuda" if values["use_gpu"] else "cpu",
                        block_seconds=values["block_seconds"],
                        input_device=input_device_indices[
                            window["input_device"].widget.current()
                        ],
                        output_device=output_device_indices[
                            window["output_device"].widget.current()
                        ],
                        passthrough_original=values["passthrough_original"],
                    ),
                )
            elif event == "stop_vc":
                if future:
                    future.cancel()
                    future = None
            elif event == "onnx_export":
                from .onnx_export import onnx_export

                try:
                    onnx_export(
                        input_path=Path(values["model_path"]),
                        output_path=Path(values["model_path"]).with_suffix(".onnx"),
                        config_path=Path(values["config_path"]),
                        device="cpu",
                    )
                except Exception as e:
                    LOG.exception(e)
            if future is not None and future.done():
                LOG.error("Error in realtime: ")
                try:
                    future.result()
                except Exception as e:
                    LOG.exception(e)
                future = None
        if future:
            future.cancel()
    window.close()
    
