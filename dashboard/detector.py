"""Car detector and cut-out without PyTorch: YOLO11m-seg exported to ONNX, with the same pre- and post-processing as Ultralytics.

detect(pixels) -> (mask, box, confidence) for the biggest car or truck, or None.
  pixels: H×W×3 uint8 RGB. mask: H×W bool. box: (x1, y1, x2, y2) in pixels, rounded.
"""
import cv2
import numpy as np

SIZE = 640
CONF = 0.25          # Ultralytics defaults
IOU = 0.7
CAR_CLASSES = (2, 7)  # COCO: car, truck


def _letterbox(img):
    h, w = img.shape[:2]
    r = min(SIZE / h, SIZE / w)
    nw, nh = round(w * r), round(h * r)
    dw, dh = (SIZE - nw) / 2, (SIZE - nh) / 2
    if (w, h) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom, left, right = round(dh - 0.1), round(dh + 0.1), round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (dw, dh)


def _nms(boxes, scores, classes):
    """Class-aware non-maximum suppression: returns the indices to keep, best first."""
    keep = []
    for c in np.unique(classes):
        idx = np.flatnonzero(classes == c)
        xywh = np.column_stack([boxes[idx, 0], boxes[idx, 1], boxes[idx, 2] - boxes[idx, 0], boxes[idx, 3] - boxes[idx, 1]])
        kept = cv2.dnn.NMSBoxes(xywh.tolist(), scores[idx].tolist(), CONF, IOU)
        keep += idx[np.array(kept, dtype=int).reshape(-1)].tolist()
    return sorted(keep, key=lambda i: -scores[i])


def detect(session, pixels):
    h0, w0 = pixels.shape[:2]
    img, r, (dw, dh) = _letterbox(pixels)
    x = np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    out, protos = session.run(None, {session.get_inputs()[0].name: x})
    pred = out[0].T                                   # 8400 × 116: xywh, 80 class scores, 32 mask coefficients
    scores_all = pred[:, 4:84]
    cls = scores_all.argmax(1)
    conf = scores_all[np.arange(len(pred)), cls]
    sel = (conf > CONF) & np.isin(cls, CAR_CLASSES)
    if not sel.any():
        return None
    pred, conf, cls = pred[sel], conf[sel], cls[sel]
    xyxy = np.column_stack([pred[:, 0] - pred[:, 2] / 2, pred[:, 1] - pred[:, 3] / 2, pred[:, 0] + pred[:, 2] / 2, pred[:, 1] + pred[:, 3] / 2])
    keep = _nms(xyxy, conf, cls)[:300]
    xyxy, conf, coef = xyxy[keep], conf[keep], pred[keep, 84:]
    # masks: coefficients × prototypes. The biggest car is picked on the coarse 160×160 masks, and only that
    # one is scaled up to the photo (scaling every detection to full size costs hundreds of MB on a large photo).
    m = (coef @ protos[0].reshape(32, -1)).reshape(-1, 160, 160)
    boxes = xyxy.copy()
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - dw) / r).clip(0, w0)
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - dh) / r).clip(0, h0)
    gx, gy = np.arange(160)[None, None, :] * 4 + 2, np.arange(160)[None, :, None] * 4 + 2   # centre of each coarse cell, in the letterboxed image
    inside = (gx >= xyxy[:, 0, None, None]) & (gx < xyxy[:, 2, None, None]) & (gy >= xyxy[:, 1, None, None]) & (gy < xyxy[:, 3, None, None])
    biggest = ((m > 0) & inside).sum(axis=(1, 2)).argmax()
    t, l = round(dh * 0.25 - 0.1), round(dw * 0.25 - 0.1)
    b, rr = 160 - round(dh * 0.25 + 0.1), 160 - round(dw * 0.25 + 0.1)
    mask = cv2.resize(m[biggest, t:b, l:rr], (w0, h0), interpolation=cv2.INTER_LINEAR) > 0
    x1, y1, x2, y2 = boxes[biggest]
    mask[:, : max(int(np.ceil(x1)), 0)] = False
    mask[:, int(np.ceil(x2)):] = False
    mask[: max(int(np.ceil(y1)), 0), :] = False
    mask[int(np.ceil(y2)):, :] = False
    return mask, tuple(boxes[biggest].round().astype(int)), float(conf[biggest])
