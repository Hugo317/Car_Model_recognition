"""Convert the trained Car Spotter models to ONNX / NumPy so the app runs without TensorFlow and PyTorch.

Usage (TensorFlow 2.18, tf2onnx and ultralytics installed; run from the repo root):
    python scripts/convert_to_onnx.py models_onnx
Writes brand.onnx, heads.npz, finetuned/<brand>.onnx (finished fine-tunes only) and yolo11m-seg.onnx.
Upload the folder to the Hugging Face Hub repo named in dashboard/recognition.py (WEIGHTS_REPO).
"""
import json, os, pathlib, shutil, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf, keras, tf2onnx

M = pathlib.Path("models")
OUT = pathlib.Path(sys.argv[1])
(OUT / "heads").mkdir(parents=True, exist_ok=True)
(OUT / "finetuned").mkdir(exist_ok=True)
SPEC = [tf.TensorSpec([None, 224, 224, 3], tf.float32, name="images")]


def to_onnx(fn, path, names):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    frozen = convert_variables_to_constants_v2(tf.function(fn, input_signature=SPEC).get_concrete_function())
    tf2onnx.convert.from_graph_def(frozen.graph.as_graph_def(), input_names=[frozen.inputs[0].name], output_names=[o.name for o in frozen.outputs],
                                   opset=17, output_path=str(path))
    print("wrote", path.name, round(path.stat().st_size / 1e6, 1), "MB", [o.name for o in frozen.outputs])


# 1) brand model: features (1280) and brand logits (33) from one pass
brand = keras.models.load_model(M / "efficientnet_b0.keras")
base = brand.get_layer("efficientnetb0")
gap, dense = brand.get_layer("global_average_pooling2d"), brand.get_layer("dense")
def brand_fn(x):
    f = gap(base(x, training=False))
    return {"features": f, "logits": dense(f)}
to_onnx(brand_fn, OUT / "brand.onnx", None)

# 2) the 33 car-model classifiers on features -> plain numpy weights
heads = {}
for p in sorted((M / "model_heads").glob("*.keras")):
    m = keras.models.load_model(p)
    d1, d2 = m.layers[0], m.layers[2]
    heads[p.stem] = [d1.kernel.numpy(), d1.bias.numpy(), d2.kernel.numpy(), d2.bias.numpy()]
    shutil.copy(M / "model_heads" / f"{p.stem}_classes.json", OUT / "heads" / f"{p.stem}_classes.json")
np.savez_compressed(OUT / "heads.npz", **{f"{b}|{i}": w for b, ws in heads.items() for i, w in enumerate(ws)})
print("heads", len(heads), round((OUT / "heads.npz").stat().st_size / 1e6, 1), "MB")

# 3) fine-tuned classifiers (full image models)
for p in sorted((M / "model_finetuned").glob("*.keras")):
    if not (M / "model_finetuned" / f"{p.stem}_done.txt").exists():
        print("skip (not done)", p.stem); continue
    m = keras.models.load_model(p)
    to_onnx(lambda x, m=m: {"logits": m(x, training=False)}, OUT / "finetuned" / f"{p.stem}.onnx", None)

# 4) the car detector: YOLO11m-seg, exported at 640×640 (dashboard/detector.py does the pre- and post-processing)
from ultralytics import YOLO
exported = YOLO(str(M / "yolo11m-seg.pt")).export(format="onnx", imgsz=640, opset=17, simplify=True, dynamic=False)
shutil.move(exported, OUT / "yolo11m-seg.onnx")
print("wrote yolo11m-seg.onnx")
