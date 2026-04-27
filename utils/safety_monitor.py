"""
Safety Monitor - Core pipeline
FIXED: Log violation IMMEDIATELY, identify face in background after
       This way alerts never stop regardless of face recognition speed
"""
import cv2
import os
import time
import base64
import threading
import numpy as np
from datetime import datetime
from collections import defaultdict

from utils.helmet_detector import HelmetDetector
from utils.face_recognizer  import FaceRecognizer
from database.models        import get_session, Violation
from utils.sms_alert        import send_sms_alert


class SafetyMonitor:
    def __init__(self, config):
        self.config  = config
        self.running = False
        self.sio     = None

        os.makedirs(config['violations_path'], exist_ok=True)
        os.makedirs(config['workers_db_path'],  exist_ok=True)
        get_session(config['database_url'])

        self.detector = HelmetDetector(
            model_path = config['model_path'],
            confidence = config.get('confidence', 0.25),
            iou        = config.get('iou', 0.45),
        )
        self.face_rec = FaceRecognizer(
            db_path   = config['workers_db_path'],
            threshold = config.get('face_threshold', 0.95),
        )

        sources          = [s.strip() for s in str(config.get('cameras', '0')).split(',')]
        self.sources     = sources
        self.cam_ids     = [f"cam_{i}" for i in range(len(sources))]
        self.raw_frames  = {}
        self.raw_lock    = threading.Lock()
        self.frames      = {}
        self.frame_lock  = threading.Lock()

        self.cooldown    = defaultdict(float)
        self.cooldown_s  = config.get('cooldown', 5)
        self.fps_map     = {}
        self.process_every = 3

        print(f"[Monitor] Ready | Cameras: {self.sources} | Cooldown: {self.cooldown_s}s")

    def set_socketio(self, sio):
        self.sio = sio
        print("[Monitor] SocketIO set")

    def start(self):
        self.running = True
        for i, src in enumerate(self.sources):
            cam_id = f"cam_{i}"
            threading.Thread(target=self._reader,    args=(src, cam_id), daemon=True).start()
            threading.Thread(target=self._processor, args=(cam_id,),     daemon=True).start()
        print(f"[Monitor] Started {len(self.sources)} camera(s)")

    def stop(self):
        self.running = False

    # ── Reader thread ─────────────────────────────────────────────────────────
    def _reader(self, source, cam_id):
        src = int(source) if str(source).isdigit() else source
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                cap.release()
                cap = cv2.VideoCapture(src)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue
            with self.raw_lock:
                self.raw_frames[cam_id] = frame

        cap.release()

    # ── Processor thread ──────────────────────────────────────────────────────
    def _processor(self, cam_id):
        frame_count = 0
        fps_t = time.time()
        fps_n = 0
        time.sleep(1)

        while self.running:
            with self.raw_lock:
                frame = self.raw_frames.get(cam_id)

            if frame is None:
                time.sleep(0.05)
                continue

            frame_count += 1

            if frame_count % self.process_every != 0:
                with self.frame_lock:
                    if cam_id not in self.frames:
                        self.frames[cam_id] = frame
                time.sleep(0.01)
                continue

            try:
                detections, annotated, violators = self.detector.detect(frame)
                self.detector.draw_stats(annotated, len(detections), len(violators))

                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cv2.putText(annotated, f"{cam_id}  {ts}",
                            (10, annotated.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

                with self.frame_lock:
                    self.frames[cam_id] = annotated

                if violators:
                    threading.Thread(
                        target=self._handle_violations,
                        args=(violators, frame.copy(), annotated.copy(), cam_id),
                        daemon=True
                    ).start()

            except Exception as e:
                print(f"[Processor] {cam_id} error: {e}")

            fps_n += 1
            if time.time() - fps_t >= 1.0:
                self.fps_map[cam_id] = round(fps_n / (time.time() - fps_t), 1)
                fps_n = 0
                fps_t = time.time()

            time.sleep(0.01)

    # ── Violation handler ─────────────────────────────────────────────────────
    def _handle_violations(self, violators, raw_frame, ann_frame, cam_id):
        for violator in violators:
            try:
                cam_key = f"cam_{cam_id}"
                now     = time.time()

                if now - self.cooldown[cam_key] < self.cooldown_s:
                    remaining = self.cooldown_s - (now - self.cooldown[cam_key])
                    print(f"[Monitor] Cooldown: {remaining:.0f}s left")
                    continue

                self.cooldown[cam_key] = now

                img_path = self._save_img(ann_frame, cam_id)

                session = get_session()
                try:
                    v = Violation(
                        worker_id       = None,
                        worker_name     = 'Unknown',
                        violation_type  = 'No Helmet',
                        camera_id       = cam_id,
                        confidence      = violator.get('confidence', 0.0),
                        face_confidence = 0.0,
                        image_path      = img_path,
                        timestamp       = datetime.now(),
                    )
                    session.add(v)
                    session.commit()
                    violation_id = v.id
                    print(f"[Monitor] VIOLATION #{violation_id} saved instantly")
                except Exception as db_err:
                    session.rollback()
                    print(f"[Monitor] DB error: {db_err}")
                    violation_id = None
                finally:
                    session.close()

                if violation_id and self.face_rec.workers:
                    threading.Thread(
                        target=self._identify_and_update,
                        args=(raw_frame.copy(), violation_id, cam_id),
                        daemon=True
                    ).start()

            except Exception as e:
                print(f"[Monitor] Handler error: {e}")

    # ── Face recognition + SMS alert ──────────────────────────────────────────
    def _identify_and_update(self, frame, violation_id, cam_id):
        """3 outer calls x 5 inner attempts = up to 15 chances to identify."""
        try:
            print(f"[Monitor] Face recognition for #{violation_id}...")
            best_ident = None
            best_conf  = 0.0

            for attempt in range(3):
                with self.raw_lock:
                    latest = self.raw_frames.get(cam_id)
                use_frame = latest.copy() if latest is not None else frame
                ident = self.face_rec.identify(use_frame)
                conf  = ident.get('confidence', 0.0)

                if ident.get('matched') and conf > best_conf:
                    best_conf  = conf
                    best_ident = ident
                    print(f"[Monitor] Call {attempt+1}: {ident['name']} {conf:.1f}%")
                    break
                else:
                    print(f"[Monitor] Call {attempt+1}: no match")
                time.sleep(1.0)

            if best_ident and best_ident.get('matched'):
                session = get_session()
                try:
                    v = session.query(Violation).get(violation_id)
                    if v:
                        v.worker_id       = best_ident['worker_id']
                        v.worker_name     = best_ident['name']
                        v.face_confidence = best_ident.get('confidence', 0.0)
                        session.commit()
                        print(f"[Monitor] #{violation_id} -> {best_ident['name']} ({best_conf:.1f}%)")

                        # ── SMS ALERT ─────────────────────────────────────
                        try:
                            from database.models import Worker
                            worker = session.query(Worker).filter(
                                Worker.worker_id == best_ident['worker_id']
                            ).first()
                            if worker and worker.phone:
                                ts = datetime.now().strftime("%d-%m-%Y %H:%M")
                                send_sms_alert(
                                    phone_number   = worker.phone,
                                    worker_name    = best_ident['name'],
                                    violation_type = v.violation_type,
                                    timestamp      = ts,
                                )
                            else:
                                print(f"[Monitor] No phone for {best_ident['name']} — SMS skipped")
                        except Exception as sms_err:
                            print(f"[Monitor] SMS error: {sms_err}")
                        # ─────────────────────────────────────────────────

                except Exception as e:
                    session.rollback()
                    print(f"[Monitor] Update error: {e}")
                finally:
                    session.close()
            else:
                print(f"[Monitor] #{violation_id} stays Unknown")

        except Exception as e:
            print(f"[Monitor] Face ID error: {e}")

    def _save_img(self, frame, cam_id):
        ts   = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        path = os.path.join(self.config['violations_path'], f"violation_{cam_id}_{ts}.jpg")
        cv2.imwrite(path, frame)
        return path

    def get_frame_b64(self, cam_id):
        with self.frame_lock:
            frame = self.frames.get(cam_id)
        if frame is None:
            return None
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buf).decode('utf-8')

    def stats(self):
        return {
            'running':            self.running,
            'camera_ids':         self.cam_ids,
            'active_cameras':     len(self.cam_ids),
            'fps':                self.fps_map,
            'registered_workers': len(self.face_rec.all_workers()),
        }