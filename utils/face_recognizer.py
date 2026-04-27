"""
Face Recognition - FIXED
Model    : VGG-Face
Threshold: 0.95 (hardcoded — cannot be overridden by config)
"""
import os
import cv2
import time
import numpy as np
import pickle
import threading
from datetime import datetime


class FaceRecognizer:
    def __init__(self, db_path, threshold=0.95):
        self.db_path      = db_path
        self.threshold    = 0.95  # hardcoded — ignore whatever config passes
        self.workers      = {}
        self._model_ready = False
        os.makedirs(db_path, exist_ok=True)
        self._load()
        print(f"[FaceRec] {len(self.workers)} workers | threshold={self.threshold}")
        if self.workers:
            print(f"[FaceRec] Workers: {[v['name'] for v in self.workers.values()]}")
        threading.Thread(target=self._preload_model, daemon=True).start()

    def _preload_model(self):
        try:
            print("[FaceRec] Preloading VGG-Face model...")
            from deepface import DeepFace
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)
            DeepFace.represent(
                img_path          = dummy,
                model_name        = 'VGG-Face',
                enforce_detection = False,
                detector_backend  = 'opencv',
            )
            self._model_ready = True
            print("[FaceRec] Model ready!")
        except Exception as e:
            print(f"[FaceRec] Preload warning: {e}")
            self._model_ready = True

    def _emb_file(self):
        return os.path.join(self.db_path, 'embeddings.pkl')

    def _load(self):
        f = self._emb_file()
        if os.path.exists(f):
            try:
                with open(f, 'rb') as fp:
                    self.workers = pickle.load(fp)
                print(f"[FaceRec] Loaded {len(self.workers)} workers from disk")
            except Exception as e:
                print(f"[FaceRec] Load error: {e}")
                self.workers = {}

    def _save(self):
        try:
            with open(self._emb_file(), 'wb') as fp:
                pickle.dump(self.workers, fp)
            print(f"[FaceRec] Saved: {[v['name'] for v in self.workers.values()]}")
        except Exception as e:
            print(f"[FaceRec] Save error: {e}")

    def _crop_face(self, img):
        try:
            gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            if len(faces) > 0:
                x, y, w, h = faces[0]
                pad = int(0.2 * w)
                x1 = max(0, x - pad);  y1 = max(0, y - pad)
                x2 = min(img.shape[1], x + w + pad)
                y2 = min(img.shape[0], y + h + pad)
                return cv2.resize(img[y1:y2, x1:x2], (224, 224))
        except Exception:
            pass
        return None

    def _get_embedding(self, img, fast=False):
        try:
            from deepface import DeepFace
            backend = 'opencv' if fast else 'ssd'
            try:
                result = DeepFace.represent(
                    img_path          = img,
                    model_name        = 'VGG-Face',
                    enforce_detection = True,
                    detector_backend  = backend,
                )
            except Exception:
                result = DeepFace.represent(
                    img_path          = img,
                    model_name        = 'VGG-Face',
                    enforce_detection = False,
                    detector_backend  = backend,
                )
            if result and len(result) > 0:
                return np.array(result[0]['embedding'])
            return None
        except Exception as e:
            print(f"[FaceRec] Embedding error: {e}")
            return None

    def register(self, worker_id, name, face_img, designation='', department=''):
        print(f"[FaceRec] Registering {name}...")
        waited = 0
        while not self._model_ready and waited < 30:
            print(f"[FaceRec] Waiting for model... {waited}s")
            time.sleep(1)
            waited += 1

        cropped = self._crop_face(face_img)
        use_img = cropped if cropped is not None else face_img
        print(f"[FaceRec] Face {'cropped OK' if cropped is not None else 'using full image'}")

        emb = self._get_embedding(use_img, fast=True)
        if emb is None:
            emb = self._get_embedding(face_img, fast=True)
        if emb is None:
            return False, "Could not detect face. Use a clear front-facing photo in good lighting."

        wdir = os.path.join(self.db_path, worker_id)
        os.makedirs(wdir, exist_ok=True)
        cv2.imwrite(os.path.join(wdir, 'face.jpg'), face_img)

        self._load()
        self.workers[worker_id] = {
            'name':        name,
            'designation': designation,
            'department':  department,
            'embedding':   emb.tolist(),
            'registered':  datetime.now().isoformat(),
        }
        self._save()
        print(f"[FaceRec] Registered {name} | total={len(self.workers)}")
        return True, f"{name} registered successfully"

    def identify(self, frame):
        self.threshold = 0.95  # always force correct threshold
        self._load()

        empty = {'matched': False, 'worker_id': None, 'name': 'Unknown', 'confidence': 0.0}
        if frame is None or not self.workers:
            return empty

        best_result = empty
        best_dist   = float('inf')

        for attempt in range(5):
            emb = self._get_embedding(frame, fast=False)
            if emb is None:
                print(f"[FaceRec] Attempt {attempt+1}: no face detected")
                time.sleep(0.3)
                continue

            for wid, data in self.workers.items():
                stored = np.array(data['embedding'])
                dist   = self._cosine_dist(emb, stored)
                print(f"[FaceRec] Attempt {attempt+1} | {data['name']}: dist={dist:.4f} (need <= {self.threshold})")

                if dist < best_dist:
                    best_dist = dist
                    if dist <= self.threshold:
                        conf = round(max(0.0, (1.0 - dist / self.threshold)) * 100, 1)
                        best_result = {
                            'matched':     True,
                            'worker_id':   wid,
                            'name':        data['name'],
                            'designation': data.get('designation', ''),
                            'department':  data.get('department', ''),
                            'confidence':  conf,
                        }
                        print(f"[FaceRec] MATCH: {data['name']} dist={dist:.4f} conf={conf}%")

            if best_result['matched'] and best_result['confidence'] >= 5.0:
                print(f"[FaceRec] Match found at attempt {attempt+1} — stopping")
                break

            time.sleep(0.3)

        if not best_result['matched']:
            print(f"[FaceRec] No match | best dist={best_dist:.4f} | threshold={self.threshold}")

        return best_result

    def _cosine_dist(self, a, b):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 1.0
        return 1.0 - float(np.dot(a, b) / (na * nb))

    def all_workers(self):
        return [
            {
                'worker_id':   wid,
                'name':        d['name'],
                'designation': d.get('designation', ''),
                'department':  d.get('department', ''),
                'registered':  d.get('registered', ''),
            }
            for wid, d in self.workers.items()
        ]

    def remove(self, worker_id):
        if worker_id in self.workers:
            del self.workers[worker_id]
            self._save()
            return True
        return False