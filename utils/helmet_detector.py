"""
Helmet Detector — updated for jomarkow/Safety-Helmet-Detection model
Model classes: {0: 'helmet', 1: 'head', 2: 'person'}

  'helmet' → person IS wearing helmet  → GREEN box ✅
  'head'   → person NOT wearing helmet → RED box  🚨 VIOLATION
  'person' → generic person            → ignored  ⬜
"""
import cv2
import os
from ultralytics import YOLO

VIOLATION_KEYWORDS = {
    'head', 'no-helmet', 'no_helmet', 'nohelmet',
    'no-hardhat', 'no_hardhat', 'nohardhat',
    'without_helmet', 'bare_head',
}

SAFE_KEYWORDS = {
    'helmet', 'hardhat', 'hard_hat', 'hard-hat',
    'helm', 'with_helmet', 'with_hardhat', 'safety_helmet',
}


class HelmetDetector:
    def __init__(self, model_path, confidence=0.30, iou=0.45):
        self.confidence = confidence
        self.iou        = iou

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"\n❌ Model not found: {model_path}\n"
                f"   Place best.pt inside the models/ folder and rename it helmet_yolov8.pt"
            )

        print(f"[Detector] Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.names = self.model.names
        print(f"[Detector] Model classes: {self.names}")

        self.violation_ids = set()
        self.safe_ids      = set()

        print("[Detector] Class mapping:")
        for cid, cname in self.names.items():
            low = cname.lower().strip()
            if low in VIOLATION_KEYWORDS:
                self.violation_ids.add(cid)
                print(f"  🚨  [{cid}] '{cname}'  →  NO HELMET  (RED box)")
            elif low in SAFE_KEYWORDS:
                self.safe_ids.add(cid)
                print(f"  ✅  [{cid}] '{cname}'  →  HELMET OK  (GREEN box)")
            else:
                print(f"  ⬜  [{cid}] '{cname}'  →  ignored")

        if not self.violation_ids:
            print(f"\n  ⚠️  WARNING: No violation class found in {self.names}")
            print("  ⚠️  Expected class named 'head' or 'no-helmet'\n")
        if not self.safe_ids:
            print(f"\n  ⚠️  WARNING: No safe class found in {self.names}")
            print("  ⚠️  Expected class named 'helmet' or 'hardhat'\n")

    def detect(self, frame):
        """
        Returns:
          detections     - all boxes
          annotated      - frame with boxes drawn
          violators      - only the no-helmet detections
        """
        results = self.model(frame, conf=self.confidence, iou=self.iou, verbose=False)

        detections = []
        violators  = []
        annotated  = frame.copy()

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf   = float(box.conf[0])
                cls_id = int(box.cls[0])
                cname  = self.names.get(cls_id, 'unknown')

                is_viol = cls_id in self.violation_ids
                is_safe = cls_id in self.safe_ids

                det = {
                    'bbox':         (x1, y1, x2, y2),
                    'confidence':   conf,
                    'class_id':     cls_id,
                    'class_name':   cname,
                    'is_violation': is_viol,
                }

                if is_viol:
                    color = (0, 0, 255)           # RED
                    label = f"NO HELMET  {conf:.0%}"
                    det['face_region'] = self._face_crop(frame, x1, y1, x2, y2)
                    violators.append(det)

                elif is_safe:
                    color = (0, 200, 0)            # GREEN
                    label = f"Helmet OK  {conf:.0%}"

                else:
                    color = (150, 150, 150)        # GREY — person/other
                    label = f"{cname}  {conf:.0%}"

                # Draw box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Label background
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 12), (x1 + tw + 6, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                detections.append(det)

        return detections, annotated, violators

    def _face_crop(self, frame, x1, y1, x2, y2):
        """Crop face region from detection box"""
        h, w = y2 - y1, x2 - x1
        pad  = int(w * 0.1)
        fy1  = max(0, int(y1 + h * 0.4) - pad)
        fy2  = min(frame.shape[0], y2 + pad)
        fx1  = max(0, x1 - pad)
        fx2  = min(frame.shape[1], x2 + pad)
        crop = frame[fy1:fy2, fx1:fx2]
        return crop if crop.size > 0 else None

    def draw_stats(self, frame, n_det, n_viol):
        """Draw stats overlay"""
        ov = frame.copy()
        cv2.rectangle(ov, (8, 8), (270, 90), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, f"Detections : {n_det}",  (16, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        vc = (0, 0, 255) if n_viol > 0 else (0, 200, 0)
        cv2.putText(frame, f"Violations : {n_viol}", (16, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, vc, 2)
        return frame
