import sys, os

# Always add project root to path — works no matter where you run from
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)  # set working directory to project root

import cv2
import base64
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_file, send_from_directory, Response
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from sqlalchemy import func, desc

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE, 'database', 'safety.db')
DB_URL   = f"sqlite:///{DB_PATH}"
MODEL    = os.path.join(BASE, 'models', 'helmet_yolov8.pt')
WORKERS  = os.path.join(BASE, 'database', 'workers_faces')
VIOLS    = os.path.join(BASE, 'database', 'violations')
FRONTEND = os.path.join(BASE, 'frontend')

for d in [os.path.dirname(DB_PATH), WORKERS, VIOLS]:
    os.makedirs(d, exist_ok=True)

if not os.path.exists(MODEL) or os.path.getsize(MODEL) < 1_000_000:
    print("[App] Model not found — downloading now...")
    sys.path.insert(0, os.path.join(BASE, 'models'))
    from download_model import download_model
    download_model()

from database.models import get_session, init_db, Worker, Violation
init_db(DB_URL)
get_session(DB_URL)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'safety-secret-key'
CORS(app, resources={r"/*": {"origins": "*"}})
sio = SocketIO(app, cors_allowed_origins="*", async_mode='threading',
               ping_timeout=120, ping_interval=10)

from utils.safety_monitor import SafetyMonitor

monitor = SafetyMonitor({
    'model_path':      MODEL,
    'confidence':      0.25,
    'iou':             0.45,
    'workers_db_path': WORKERS,
    'violations_path': VIOLS,
    'database_url':    DB_URL,
    'cameras':         '0',
    'cooldown':        5,
    'face_threshold':  0.30,
})
monitor.set_socketio(sio)

@app.route('/')
def index():
    return send_from_directory(FRONTEND, 'dashboard.html')

@sio.on('connect')
def on_connect():
    print(f"[Socket] Client connected")
    emit('connected', {'status': 'ok'})

@sio.on('disconnect')
def on_disconnect():
    print(f"[Socket] Client disconnected")

@sio.on('ping_client')
def on_ping():
    emit('pong_server', {'time': datetime.now().isoformat()})

def _mjpeg(cam_id):
    import time
    while True:
        b64 = monitor.get_frame_b64(cam_id)
        if b64:
            data = base64.b64decode(b64)
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + data + b'\r\n'
        else:
            time.sleep(0.05)

@app.route('/api/stream/<cam_id>')
def stream(cam_id):
    return Response(_mjpeg(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    s = get_session()
    try:
        today_start = datetime.combine(datetime.now().date(), datetime.min.time())
        today  = s.query(Violation).filter(Violation.timestamp >= today_start).count()
        total  = s.query(Violation).count()
        unique = s.query(func.count(func.distinct(Violation.worker_name)))\
                   .filter(Violation.timestamp >= today_start).scalar() or 0
        hourly = s.query(
            func.strftime('%H', Violation.timestamp).label('hour'),
            func.count(Violation.id).label('count')
        ).filter(Violation.timestamp >= datetime.now() - timedelta(hours=24))\
         .group_by('hour').all()
        st = monitor.stats()
        return jsonify({
            'today_violations':   today,
            'total_violations':   total,
            'unique_today':       unique,
            'registered_workers': st['registered_workers'],
            'active_cameras':     st['active_cameras'],
            'camera_ids':         st['camera_ids'],
            'fps':                st['fps'],
            'hourly':             [{'hour': h.hour, 'count': h.count} for h in hourly],
        })
    finally:
        s.close()

@app.route('/api/violations')
def violations():
    s = get_session()
    try:
        page     = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        name     = request.args.get('name', '')
        resolved = request.args.get('resolved', '')
        q = s.query(Violation)
        if name:
            q = q.filter(Violation.worker_name.ilike(f'%{name}%'))
        if resolved != '':
            q = q.filter(Violation.is_resolved == (resolved == 'true'))
        total = q.count()
        rows  = q.order_by(desc(Violation.timestamp))\
                  .offset((page - 1) * per_page).limit(per_page).all()
        return jsonify({
            'violations': [v.to_dict() for v in rows],
            'total':      total,
            'page':       page,
            'pages':      max(1, (total + per_page - 1) // per_page),
        })
    finally:
        s.close()

@app.route('/api/violations/recent')
def recent_violations():
    s = get_session()
    try:
        rows = s.query(Violation).order_by(desc(Violation.timestamp)).limit(10).all()
        return jsonify([v.to_dict() for v in rows])
    finally:
        s.close()

@app.route('/api/violations/<int:vid>', methods=['PATCH'])
def patch_violation(vid):
    s = get_session()
    try:
        v = s.query(Violation).get(vid)
        if not v:
            return jsonify({'error': 'not found'}), 404
        d = request.json
        if 'is_resolved' in d:
            v.is_resolved = d['is_resolved']
        if 'notes' in d:
            v.notes = d['notes']
        s.commit()
        return jsonify(v.to_dict())
    finally:
        s.close()

@app.route('/api/violations/<int:vid>/image')
def violation_image(vid):
    s = get_session()
    try:
        v = s.query(Violation).get(vid)
        if not v or not v.image_path or not os.path.exists(v.image_path):
            return jsonify({'error': 'image not found'}), 404
        return send_file(v.image_path, mimetype='image/jpeg')
    finally:
        s.close()

@app.route('/api/workers', methods=['GET'])
def get_workers():
    s = get_session()
    try:
        ws = s.query(Worker).filter(Worker.is_active == True).all()
        return jsonify([w.to_dict() for w in ws])
    finally:
        s.close()

@app.route('/api/workers', methods=['POST'])
def add_worker():
    s = get_session()
    try:
        wid  = request.form.get('worker_id', '').strip()
        name = request.form.get('name', '').strip()
        des  = request.form.get('designation', '')
        dept = request.form.get('department', '')
        ph   = request.form.get('phone', '')
        if not wid or not name:
            return jsonify({'error': 'worker_id and name required'}), 400
        if s.query(Worker).filter(Worker.worker_id == wid).first():
            return jsonify({'error': 'Worker ID already exists'}), 409
        face_img = None
        if 'face_image' in request.files:
            file = request.files['face_image']
            data = file.read()
            print(f"[App] Received face image: {len(data)} bytes")
            if len(data) > 0:
                arr      = np.frombuffer(data, np.uint8)
                face_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if face_img is not None:
                    print(f"[App] Face image decoded: {face_img.shape}")
                else:
                    print("[App] ERROR: cv2.imdecode returned None!")
                    return jsonify({'error': 'Could not decode image. Use JPG or PNG.'}), 400
            else:
                print("[App] ERROR: Empty file uploaded!")
                return jsonify({'error': 'Empty file uploaded'}), 400
        else:
            print("[App] No face_image in request — registering without face")

        if face_img is not None:
            ok, msg = monitor.face_rec.register(wid, name, face_img, des, dept)
            print(f"[App] Registration result: ok={ok}, msg={msg}")
            if not ok:
                return jsonify({'error': msg}), 400
        else:
            return jsonify({'error': 'Please upload a face photo to register worker'}), 400
        w = Worker(
            worker_id=wid, name=name, designation=des,
            department=dept, phone=ph,
            face_path=os.path.join(WORKERS, wid, 'face.jpg')
        )
        s.add(w)
        s.commit()
        return jsonify({'message': 'Worker registered', 'worker': w.to_dict()}), 201
    except Exception as e:
        s.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        s.close()

@app.route('/api/workers/<wid>', methods=['DELETE'])
def del_worker(wid):
    s = get_session()
    try:
        w = s.query(Worker).filter(Worker.worker_id == wid).first()
        if not w:
            return jsonify({'error': 'not found'}), 404
        w.is_active = False
        monitor.face_rec.remove(wid)
        s.commit()
        return jsonify({'message': 'removed'})
    finally:
        s.close()

@app.route('/api/reports/daily')
def daily():
    s = get_session()
    try:
        rows = s.query(
            func.strftime('%Y-%m-%d', Violation.timestamp).label('date'),
            func.count(Violation.id).label('count')
        ).filter(Violation.timestamp >= datetime.now() - timedelta(days=30))\
         .group_by('date').order_by('date').all()
        return jsonify([{'date': r.date, 'count': r.count} for r in rows])
    finally:
        s.close()

@app.route('/api/reports/top-violators')
def top_violators():
    s = get_session()
    try:
        rows = s.query(
            Violation.worker_name,
            func.count(Violation.id).label('count')
        ).filter(
            Violation.timestamp >= datetime.now() - timedelta(days=7),
            Violation.worker_name != 'Unknown'
        ).group_by(Violation.worker_name)\
         .order_by(desc('count')).limit(10).all()
        return jsonify([{'name': r.worker_name, 'count': r.count} for r in rows])
    finally:
        s.close()

@app.route('/api/system/start', methods=['POST'])
def sys_start():
    if not monitor.running:
        monitor.start()
    return jsonify({'status': 'running'})

@app.route('/api/system/stop', methods=['POST'])
def sys_stop():
    monitor.stop()
    return jsonify({'status': 'stopped'})

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'time': datetime.now().isoformat()})

if __name__ == '__main__':
    print("=" * 55)
    print("  Construction Safety Monitoring System")
    print("=" * 55)
    print("  Dashboard : http://localhost:5000")
    print("=" * 55)
    monitor.start()
    sio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
