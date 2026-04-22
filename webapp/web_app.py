from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms

from data_loader import depth_image_to_tensor
from model import CalorieRegressor

DEPTH_MODE_MIDAS = "Auto from RGB (MiDaS)"
DEPTH_MODE_HEURISTIC = "Auto from RGB (Heuristic)"
DEPTH_MODE_REAL_UPLOAD = "Upload real depth image"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web demo for calorie prediction.")
    parser.add_argument(
        "--checkpoint_rgb",
        type=str,
        default="",
        help="Checkpoint path for RGB mode.",
    )
    parser.add_argument(
        "--checkpoint_rgbd",
        type=str,
        default="",
        help="Checkpoint path for RGB-D mode.",
    )
    # Backward compatibility for old single-checkpoint launch style.
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--mode", type=str, choices=["rgb", "rgbd"], default="rgb")
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Fallback if checkpoint has no image_size; otherwise training value from checkpoint is used.",
    )
    parser.add_argument(
        "--max_depth_units",
        type=float,
        default=4000.0,
        help="Fallback if checkpoint has no max_depth_units; otherwise training value from checkpoint is used.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--auto_depth_backend",
        type=str,
        choices=["midas", "heuristic"],
        default="midas",
        help="Backend for auto depth when depth image is missing in RGB-D mode.",
    )
    parser.add_argument("--cls_top_k", type=int, default=5, help="Top-K class candidates to show.")
    parser.add_argument(
        "--cls_conf_threshold",
        type=float,
        default=0.10,
        help="Confidence threshold for showing multiple category candidates.",
    )
    return parser.parse_args()


class Predictor:
    def __init__(
        self,
        checkpoint_path: str,
        mode: str,
        image_size: int,
        max_depth_units: float,
        auto_depth_backend: str,
        cls_top_k: int,
        cls_conf_threshold: float,
    ):
        self.mode = mode
        self.auto_depth_backend = auto_depth_backend
        self.cls_top_k = cls_top_k
        self.cls_conf_threshold = cls_conf_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        ci = ckpt.get("image_size")
        if ci is not None:
            ci = int(ci)
            if ci != image_size:
                print(f"[info] image_size: using {ci} from checkpoint (CLI was {image_size}).")
            image_size = ci
        cd = ckpt.get("max_depth_units")
        if cd is not None:
            cd = float(cd)
            if cd != max_depth_units:
                print(f"[info] max_depth_units: using {cd} from checkpoint (CLI was {max_depth_units}).")
            max_depth_units = cd

        self.image_size = image_size
        self.max_depth_units = max_depth_units
        self.use_log_target = bool(ckpt.get("use_log_target", True))
        self.food101_classes = ckpt.get("food101_classes", [])
        ckpt_mode = ckpt.get("mode", mode)
        if ckpt_mode != mode:
            raise ValueError(f"Checkpoint mode is {ckpt_mode}, but --mode is {mode}.")

        num_classes = len(self.food101_classes) if ckpt.get("has_classifier", False) else 0
        self.model = CalorieRegressor(
            mode=mode, pretrained=False, num_classes=num_classes
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.depth_resize = transforms.Resize((self.image_size, self.image_size))
        self.midas_model = None
        self.midas_transform = None
        if self.mode == "rgbd" and self.auto_depth_backend == "midas":
            self._try_load_midas()

    def _try_load_midas(self) -> None:
        try:
            self.midas_model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
            self.midas_model.to(self.device).eval()
            transforms_mod = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.midas_transform = transforms_mod.dpt_transform
            print("Loaded MiDaS auto-depth backend.")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Could not load MiDaS, fallback to heuristic depth. Error: {exc}")
            self.midas_model = None
            self.midas_transform = None

    def _prepare_rgb(self, rgb_img: Image.Image) -> Image.Image:
        rgb_img = ImageOps.exif_transpose(rgb_img)
        return rgb_img.convert("RGB")

    def _build_rgb_input(self, rgb_img: Image.Image) -> torch.Tensor:
        """Convert a user-uploaded RGB image to model-ready tensor (NCHW)."""
        rgb_img = self._prepare_rgb(rgb_img)
        return self.rgb_transform(rgb_img).unsqueeze(0).to(self.device)

    def _build_rgbd_input(self, rgb_img: Image.Image, depth_t: torch.Tensor) -> torch.Tensor:
        """Build model-ready RGB-D input from already prepared depth tensor."""
        rgb_img = self._prepare_rgb(rgb_img)
        rgb_t = self.rgb_transform(rgb_img)
        x = torch.cat([rgb_t, depth_t], dim=0).unsqueeze(0).to(self.device)
        return x

    def _heuristic_depth_from_rgb(self, rgb_img: Image.Image) -> torch.Tensor:
        # Fallback when user has no true depth: use luminance as pseudo-depth.
        gray = ImageOps.grayscale(rgb_img)
        gray = self.depth_resize(gray)
        depth_np = np.array(gray).astype(np.float32) / 255.0
        return torch.from_numpy(depth_np).unsqueeze(0)

    def _midas_depth_from_rgb(self, rgb_img: Image.Image) -> torch.Tensor:
        if self.midas_model is None or self.midas_transform is None:
            return self._heuristic_depth_from_rgb(rgb_img)

        rgb_np = np.array(rgb_img)
        input_batch = self.midas_transform(rgb_np).to(self.device)
        with torch.no_grad():
            pred = self.midas_model(input_batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=(self.depth_resize.size[0], self.depth_resize.size[1]),
                mode="bicubic",
                align_corners=False,
            ).squeeze(1).squeeze(0)
        pred = pred - pred.min()
        denom = pred.max() - pred.min()
        if float(denom) > 1e-6:
            pred = pred / denom
        pred = torch.clamp(pred, 0.0, 1.0).cpu()
        return pred.unsqueeze(0)

    def _auto_depth_from_rgb(self, rgb_img: Image.Image, backend: str | None = None) -> torch.Tensor:
        backend_name = backend or self.auto_depth_backend
        if backend_name == "midas":
            return self._midas_depth_from_rgb(rgb_img)
        return self._heuristic_depth_from_rgb(rgb_img)

    def _depth_to_tensor(self, depth_img: Image.Image) -> torch.Tensor:
        depth_img = ImageOps.exif_transpose(depth_img)
        return depth_image_to_tensor(
            depth_img=depth_img,
            resize_transform=self.depth_resize,
            max_depth_units=self.max_depth_units,
        )

    def _decode_prediction(self, pred: torch.Tensor) -> float:
        if self.use_log_target:
            pred = torch.expm1(torch.clamp(pred, max=12.0))
        pred = torch.clamp(pred, min=0.0)
        return float(pred.item())

    def _format_calories_and_class(self, x: torch.Tensor) -> str:
        with torch.no_grad():
            pred = self.model(x)
        kcal = self._decode_prediction(pred)
        cls_text = self._classification_result(x)
        return f"Predicted calories: {kcal:.2f} kcal\n{cls_text}"

    def _classification_result(self, x: torch.Tensor) -> str:
        if not self.model.has_classifier or not self.food101_classes:
            return "Food category: unavailable (classifier head not trained)."
        with torch.no_grad():
            logits = self.model.classify(x)
            probs = torch.softmax(logits, dim=1)
            top_k = min(self.cls_top_k, probs.shape[1])
            confs, cls_indices = torch.topk(probs, k=top_k, dim=1)

        conf_list = confs.squeeze(0).tolist()
        idx_list = cls_indices.squeeze(0).tolist()
        kept = []
        for conf, cls_idx in zip(conf_list, idx_list):
            if conf >= self.cls_conf_threshold:
                kept.append((self.food101_classes[int(cls_idx)], float(conf)))

        if not kept:
            kept = [(self.food101_classes[int(idx_list[0])], float(conf_list[0]))]

        lines = ["Food category candidates:"]
        for name, conf in kept:
            lines.append(f"- {name}: {conf:.3f}")
        return "\n".join(lines)

    def _depth_tensor_to_preview(
        self,
        depth_t: torch.Tensor,
        invert_preview: bool = False,
        contrast: float = 1.0,
    ) -> Image.Image:
        depth_np = depth_t.squeeze(0).detach().cpu().numpy()
        depth_np = np.clip(depth_np, 0.0, 1.0)
        if invert_preview:
            depth_np = 1.0 - depth_np
        if contrast > 1.0:
            depth_np = np.clip((depth_np - 0.5) * contrast + 0.5, 0.0, 1.0)
        depth_u8 = (depth_np * 255.0).astype(np.uint8)
        return Image.fromarray(depth_u8, mode="L")

    def predict_rgb(self, rgb_img: Image.Image) -> str:
        if rgb_img is None:
            return "Please upload an RGB image."
        x = self._build_rgb_input(rgb_img)
        return self._format_calories_and_class(x)

    def predict_rgbd(
        self,
        rgb_img: Image.Image,
        depth_source_mode: str,
        depth_img: Image.Image | None,
        invert_preview: bool,
        depth_contrast: float,
    ):
        if rgb_img is None:
            return "Please upload an RGB image.", None
        try:
            rgb_prepared = self._prepare_rgb(rgb_img)
            if depth_source_mode == DEPTH_MODE_MIDAS:
                depth_t = self._auto_depth_from_rgb(rgb_prepared, backend="midas")
            elif depth_source_mode == DEPTH_MODE_HEURISTIC:
                depth_t = self._auto_depth_from_rgb(rgb_prepared, backend="heuristic")
            elif depth_source_mode == DEPTH_MODE_REAL_UPLOAD:
                if depth_img is None:
                    raise ValueError("Please upload a real depth image in RGB-D mode.")
                depth_t = self._depth_to_tensor(depth_img)
            else:
                raise ValueError(f"Unsupported depth mode: {depth_source_mode}")
            x = self._build_rgbd_input(rgb_prepared, depth_t)
        except ValueError as exc:
            return str(exc), None
        text = self._format_calories_and_class(x)
        depth_preview = self._depth_tensor_to_preview(
            depth_t,
            invert_preview=invert_preview,
            contrast=depth_contrast,
        )
        return text, depth_preview


class AppRuntime:
    def __init__(self, rgb_predictor: Predictor | None, rgbd_predictor: Predictor | None):
        self.rgb_predictor = rgb_predictor
        self.rgbd_predictor = rgbd_predictor

    def mode_ui_updates(self, mode: str, depth_mode: str):
        is_rgbd = mode == "rgbd"
        show_real_depth_input = is_rgbd and (depth_mode == DEPTH_MODE_REAL_UPLOAD)
        return (
            gr.update(visible=is_rgbd),  # depth_mode_dd
            gr.update(visible=show_real_depth_input),  # depth_input
            gr.update(visible=is_rgbd),  # invert_preview
            gr.update(visible=is_rgbd),  # depth_contrast
            gr.update(visible=is_rgbd),  # depth_preview
        )

    def depth_mode_ui_updates(self, depth_mode: str):
        needs_depth_upload = depth_mode == DEPTH_MODE_REAL_UPLOAD
        return (gr.update(visible=needs_depth_upload),)

    def predict(
        self,
        mode: str,
        rgb_img: Image.Image,
        depth_source_mode: str,
        depth_img: Image.Image | None,
        invert_preview: bool,
        depth_contrast: float,
    ):
        if mode == "rgb":
            if self.rgb_predictor is None:
                return "RGB checkpoint is not loaded.", None
            return self.rgb_predictor.predict_rgb(rgb_img), None
        if self.rgbd_predictor is None:
            return "RGB-D checkpoint is not loaded.", None
        return self.rgbd_predictor.predict_rgbd(
            rgb_img=rgb_img,
            depth_source_mode=depth_source_mode,
            depth_img=depth_img,
            invert_preview=invert_preview,
            depth_contrast=depth_contrast,
        )


def main() -> None:
    args = parse_args()
    rgb_ckpt = args.checkpoint_rgb.strip()
    rgbd_ckpt = args.checkpoint_rgbd.strip()
    if not rgb_ckpt and not rgbd_ckpt:
        if not args.checkpoint_path:
            raise ValueError(
                "Set at least one of --checkpoint_rgb / --checkpoint_rgbd "
                "(or use legacy --checkpoint_path with --mode)."
            )
        if args.mode == "rgb":
            rgb_ckpt = args.checkpoint_path
        else:
            rgbd_ckpt = args.checkpoint_path

    rgb_predictor = None
    if rgb_ckpt:
        if not Path(rgb_ckpt).exists():
            raise FileNotFoundError(f"RGB checkpoint not found: {rgb_ckpt}")
        rgb_predictor = Predictor(
            checkpoint_path=rgb_ckpt,
            mode="rgb",
            image_size=args.image_size,
            max_depth_units=args.max_depth_units,
            auto_depth_backend=args.auto_depth_backend,
            cls_top_k=args.cls_top_k,
            cls_conf_threshold=args.cls_conf_threshold,
        )

    rgbd_predictor = None
    if rgbd_ckpt:
        if not Path(rgbd_ckpt).exists():
            raise FileNotFoundError(f"RGB-D checkpoint not found: {rgbd_ckpt}")
        rgbd_predictor = Predictor(
            checkpoint_path=rgbd_ckpt,
            mode="rgbd",
            image_size=args.image_size,
            max_depth_units=args.max_depth_units,
            auto_depth_backend=args.auto_depth_backend,
            cls_top_k=args.cls_top_k,
            cls_conf_threshold=args.cls_conf_threshold,
        )

    app = AppRuntime(rgb_predictor=rgb_predictor, rgbd_predictor=rgbd_predictor)
    mode_choices = []
    if rgb_predictor is not None:
        mode_choices.append("rgb")
    if rgbd_predictor is not None:
        mode_choices.append("rgbd")
    if not mode_choices:
        raise ValueError("No available predictor loaded.")
    default_mode = mode_choices[0]

    with gr.Blocks(title="Nutrition5k Calorie Predictor") as demo:
        gr.Markdown("## Nutrition5k Calorie Predictor")
        gr.Markdown(
            "Upload an original food photo. The app automatically preprocesses it "
            "to model input format. In RGB-D mode, it also builds depth input "
            "using one of three dropdown methods."
        )
        gr.Markdown(
            "- **RGB mode**: uploaded image is automatically preprocessed.\n"
            "- **RGB-D mode depth options**:\n"
            f"  1) `{DEPTH_MODE_MIDAS}`\n"
            f"  2) `{DEPTH_MODE_HEURISTIC}`\n"
            f"  3) `{DEPTH_MODE_REAL_UPLOAD}`"
        )
        mode_dd = gr.Dropdown(
            choices=mode_choices,
            value=default_mode,
            label="Inference Mode",
            info="Choose model mode in UI (no terminal mode switch needed).",
        )
        rgb_input = gr.Image(type="pil", label="Original Food Image (JPG/PNG)")
        depth_mode_dd = gr.Dropdown(
            choices=[DEPTH_MODE_MIDAS, DEPTH_MODE_HEURISTIC, DEPTH_MODE_REAL_UPLOAD],
            value=DEPTH_MODE_MIDAS,
            label="RGB-D Depth Source",
            info="Choose how depth is generated for RGB-D inference.",
            visible=(default_mode == "rgbd"),
        )
        depth_input = gr.Image(
            type="pil",
            label="Real Depth Image (PNG preferred)",
            visible=False,
        )
        out = gr.Textbox(label="Result")
        depth_preview = gr.Image(
            type="pil",
            label="Depth Preview",
            interactive=False,
            visible=(default_mode == "rgbd"),
        )
        invert_preview = gr.Checkbox(
            value=False, label="Invert Depth Preview", visible=(default_mode == "rgbd")
        )
        depth_contrast = gr.Slider(
            minimum=1.0,
            maximum=3.0,
            value=1.5,
            step=0.1,
            label="Depth Preview Contrast",
            visible=(default_mode == "rgbd"),
        )
        mode_dd.change(
            fn=app.mode_ui_updates,
            inputs=[mode_dd, depth_mode_dd],
            outputs=[depth_mode_dd, depth_input, invert_preview, depth_contrast, depth_preview],
        )
        depth_mode_dd.change(
            fn=app.depth_mode_ui_updates,
            inputs=[depth_mode_dd],
            outputs=[depth_input],
        )
        btn = gr.Button("Predict")
        btn.click(
            fn=app.predict,
            inputs=[mode_dd, rgb_input, depth_mode_dd, depth_input, invert_preview, depth_contrast],
            outputs=[out, depth_preview],
        )

    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
