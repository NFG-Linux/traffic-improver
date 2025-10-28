import os, re, glob, cv2, math, csv
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from ultralytics import YOLO

###END IMPORTS

model_dir = "./yolo_models"
model_type = "l"
model = YOLO(f"{model_dir}/yolov8{model_type}.pt")
conf_val = 0.35
imgsz_val = 960
iou_val = 0.45
cur_dir = os.getcwd()
out_dir = "for_eval"
cam_list_file = "list_cam.txt"
eval_dir = os.path.join(cur_dir, "eval")
train_gt_file = os.path.join(eval_dir, "ground_truth_train.txt")
val_gt_file = os.path.join(eval_dir, "ground_truth_validation.txt")
framenum_dir = os.path.join(cur_dir, "cam_framenum")
loc_dir = os.path.join(cur_dir, "cam_loc")
timestamp_dir = os.path.join(cur_dir, "cam_timestamp")
thresh = 0.7
max_miss = 15
e_mu = None
e_p = None
vehicle_cls = {1, 2, 3, 5, 7}
_CAM_RE = re.compile(r"(?:^|[\\/])c(\d{3})(?:[\\/]|$)", re.IGNORECASE)
_DSTYPE_RE = re.compile(r"(?:^|[\\/])(train|validation|test)(?:[\\/]|$)", re.IGNORECASE)
_SCENE_RE = re.compile(r"(?:^|[\\/])(S\d{2})(?:[\\/]|$)", re.IGNORECASE)

###END GLOBAL VARIABLES

@dataclass
class Track:
    tid: int
    box: Tuple[int,int,int,int]
    score: float
    cls: int
    last_frame: int
    misses: int = 0
    frames: List[int] = field(default_factory=list)
    boxes: List[Tuple[int,int,int,int]] = field(default_factory=list)
    emb: Optional[np.ndarray] = None
    
    def update(self, frame1: int, box: Tuple[int,int,int,int], score: float, cls: int, patch: Optional[np.ndarray], alpha: float = 0.3):
        self.box = box
        self.score = score
        self.cls = cls
        self.last_frame = frame1
        self.misses = 0
        self.frames.append(frame1)
        self.boxes.append(box)
        if patch is not None:
            feature = embed_reid(patch)
            if feature is not None:
                feature = apply_whitener(feature)
                if self.emb is None:
                    self.emb = feature
                else:
                    self.emb = (1 - alpha) * self.emb + alpha * feature
                    
@dataclass
class Detections:
    cam_id: int
    local_id: int
    frames: List[int]
    boxes: List[Tuple[int,int,int,int]]
    emb: Optional[np.ndarray]
    
    @property
    def start(self) -> int:
        return self.frames[0] if self.frames else 1
    
    @property
    def end(self) -> int:
        return self.frames[-1] if self.frames else 1
    
    def center_end(self) -> Tuple[float,float]:
        if not self.boxes:
            return (0.0, 0.0)
        x,y,w,h = self.boxes[-1]
        return (x + w/2, y + h/2)
    
class TrackIoU:
    def __init__(self, iou_thresh: float, max_miss: int):
        self.iou_thresh = iou_thresh
        self.max_miss = max_miss
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}
        self.gone: Dict[int, Track] = {}
        
    def step(self, frame1: int, frame_img: np.ndarray, dets: List[Tuple[Tuple[int,int,int,int], float, int]]) -> Dict[int, Track]:
        notmatched = set(self.tracks.keys())
        matches = []
        sort_dets = sorted(dets, key=lambda d: d[1], reverse=True)
        
        for di, (dbox, dscore, dcls) in enumerate(sort_dets):
            best_tid = None
            best_iou = self.iou_thresh
            for tid in list(notmatched):
                iou = xywh(self.tracks[tid].box, dbox)
                if iou >= best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_tid is not None:
                matches.append((best_tid, di))
                notmatched.discard(best_tid)
                    
        for tid, di in matches:
            dbox, dscore, dcls = sort_dets[di]
            patch = crop(frame_img, dbox)
            self.tracks[tid].update(frame1, dbox, dscore, dcls, patch)
            
        assigned_di = {di for _, di in matches}
        for di, (dbox, dscore, dcls) in enumerate(sort_dets):
            if di in assigned_di:
                continue
            tid = self.next_id
            self.next_id += 1
            patch = crop(frame_img, dbox)
            tr = Track(tid=tid, box=dbox, score=dscore, cls=dcls, last_frame=frame1)
            tr.update(frame1, dbox, dscore, dcls, patch)
            self.tracks[tid] = tr
            
        del_list = []
        for tid in list(notmatched):
            tr = self.tracks[tid]
            tr.misses += 1
            if tr.misses > self.max_miss:
                del_list.append(tid)
        for tid in del_list:
            self.gone[tid] = self.tracks[tid]
            del self.tracks[tid]
            
        return self.tracks
    
    def finish(self):
        for tid, tr in list(self.tracks.items()):
            self.gone[tid] = tr
        self.tracks.clear()
    
@dataclass
class Result:
    detections: Dict[int, 'Detections']
    width: int
    height: int
    fps: float

###END CLASSES

def xywh(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = aw * ah + bw * bh - inter
    return inter / (ua + 1e-9)

def cosine_sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return -1.0
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))

def box_frame_bind(x:int, y:int, w:int, h:int, W:int, H:int) -> Tuple[int,int,int,int]:
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h

#cam list
def cam_list(cam_list_file: str):
    cams = []
    with open(cam_list_file, "r") as f:
        rows = [rw.strip() for rw in f if rw.strip()]
        
    for cam_dir in rows:
        cam_dir = os.path.abspath(cam_dir)
        
        cam = _CAM_RE.search(cam_dir)
        cam_id = int(cam.group(1))
        cam_str = f"c{cam_id:03d}"
        dstype = _DSTYPE_RE.search(cam_dir)
        dstype_id = dstype.group(1).lower()
        scene = _SCENE_RE.search(cam_dir)
        scene_id = scene.group(1).upper()
        
        vid_path = os.path.join(cam_dir, "vdo.avi")
        roi_path = os.path.join(cam_dir, "roi.jpg")
        gt_path = os.path.join(cam_dir, "gt", "gt.txt")
        det_dir = os.path.join(cam_dir, "det")
        mtmc_dir = os.path.join(cam_dir, "mtsc")
        segm_dir = os.path.join(cam_dir, "segm", "segm_mask_rcnn.txt")
        calibrate_dir = os.path.join(cam_dir, "calibration.txt")
        
        meta = {
            "cam_dir": cam_dir,
            "cam_id": cam_id,
            "cam_str": cam_str,
            "dstype": dstype_id,
            "scene": scene_id,
            "roi": roi_path,
            "gt": gt_path,
            "det_dir": det_dir,
            "mtmc_dir": mtmc_dir,
            "segm_dir": segm_dir,
            "calibration": calibrate_dir
        }

        cams.append((cam_id, vid_path, meta))

    return cams

def parse_gt(cam_mapper, gt_file, max_per_id=40):
    rows_by_cam = defaultdict(lambda: defaultdict(list))
    with open(gt_file, "r", newline="") as f:
        for lineno, rw in enumerate(f, 1):
            s = rw.strip()
            if not s:
                continue
            parts = re.split(r"[\s]+", s)
            cam_id = int(parts[0])
            gid = int(parts[1])
            fid = int(parts[2])
            x = int(float(parts[3]))
            y = int(float(parts[4]))
            w = int(float(parts[5]))
            h = int(float(parts[6]))
            rows_by_cam[cam_id][fid].append((gid, x, y, w, h))

    X_list = []# features
    y_list = []# GID
    count_per_id = defaultdict(int)

    for cam_id, frames in rows_by_cam.items():
        if cam_id not in cam_mapper: 
            continue
        cap = cv2.VideoCapture(cam_mapper[cam_id]["vdo"])
        cur = -1
        for fid in sorted(frames.keys()):
            while cur < fid - 1:
                ret = cap.read()[0]
                if not ret: break
                cur += 1
            ret, frame = cap.read()
            if not ret: break
            cur += 1
            for gid, x0, y0, w0, h0 in frames[fid]:
                if count_per_id[gid] >= max_per_id:
                    continue
                x2, y2 = max(0, x0), max(0, y0)
                crop = frame[y2:y2+h0, x2:x2+w0]
                f = embed_reid(crop)
                if f is None: 
                    continue
                X_list.append(f)
                y_list.append(gid)
                count_per_id[gid] += 1
        cap.release()

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int32)
    return X, y

def whitener(X, out_dim=128):
    mu = X.mean(axis=0, keepdims=False)
    Xc = X - mu
    C = (Xc.T @ Xc) / max(1, Xc.shape[0]-1)
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    keep = min(out_dim, eigvecs.shape[1])
    P = eigvecs[:, :keep] @ np.diag(1.0 / np.sqrt(eigvals[:keep] + 1e-8))
    return mu.astype(np.float32), P.astype(np.float32)

def load_whitener(path: str):
    global e_mu, e_p
    if os.path.exists(path):
        z = np.load(path)
        e_mu = z["mu"].astype(np.float32).ravel()
        e_p = z["P"].astype(np.float32)
        
def apply_whitener(feature: np.ndarray) -> np.ndarray:
    if e_mu is None or e_p is None:
        return feature.astype(np.float32).ravel()
    f = (feature.astype(np.float32).ravel() - e_mu) @ e_p
    f = f / (np.linalg.norm(f) + 1e-9)
    return f.astype(np.float32)

def crop(frame: np.ndarray, box: Tuple[int,int,int,int]) -> np.ndarray:
    h, w = frame.shape[:2]
    x, y, bw, bh = box
    x2, y2 = x + bw, y + bh
    x, y = max(0, x), max(0, y)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x or y2 <= y:
        return None
    return frame[y:y2, x:x2]

def embed_reid(img: np.ndarray) -> Optional[np.ndarray]:
    if img is None or img.size == 0:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_bins, s_bins, v_bins = 16, 16, 8
    chist = cv2.calcHist([hsv], [0,1,2], None, [h_bins, s_bins, v_bins], [0,180, 0,256, 0,256]).astype(np.float32)
    chist = cv2.normalize(chist, None).flatten()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    
    m = np.percentile(mag, 99.5)
    if m > 0:
        mag = np.clip(mag, 0, m) / m
    ghist = cv2.calcHist([mag], [0], None, [32], [0, 1]).astype(np.float32).flatten()
    ghist = cv2.normalize(ghist, None).flatten()
    
    feature = np.concatenate([chist, ghist], axis=0)
    feature = np.sign(feature) * np.sqrt(np.abs(feature))
    normal = np.linalg.norm(feature) + 1e-9
    return (feature / normal).astype(np.float32)

def camera(cam_id: int, vid_path: str, model: YOLO, conf: float, imgsz: int, iou_val: float, max_miss: int, draw: bool,
           draw_path: Optional[str], roi_mask: np.ndarray) -> Result:
    cap = cv2.VideoCapture(vid_path)
    
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    writer = None
    if draw and draw_path:
        writer = cv2.VideoWriter(draw_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
        
    tracker = TrackIoU(iou_thresh=iou_val, max_miss=max_miss)
    
    frame0 = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame0 += 1
        frame1 = frame0 + 1
        
        yolo = model(frame, conf=conf, imgsz=imgsz)[0]
        data = yolo.boxes.data.cpu().numpy() if yolo.boxes is not None else np.empty((0,6), dtype=np.float32)
        
        dets: List[Tuple[Tuple[int,int,int,int], float, int]] = []
        if data.size > 0:
            xyxy = data[:, :4].astype(int)
            scores = data[:, 4]
            cls_ids = data[:, 5].astype(int)
            for (x1, y1, x2, y2), sc, cid in zip(xyxy, scores, cls_ids):
                if cid not in vehicle_cls:
                    continue
                x = int(x1); y = int(y1); w = int(x2 - x1); h = int(y2 - y1)
                x, y, w, h = box_frame_bind(x, y, w, h, W, H)
                #ROI
                cx, cy = x + w // 2, y + h // 2
                if roi_mask[min(max(cy, 0), roi_mask.shape[0]-1), min(max(cx, 0), roi_mask.shape[1]-1)] == 0:
                    continue
                #need to edit to tune, per camera bounds
                area = w * h
                ar = w / float(h + 1e-6)
                if not (100 <= area <= 0.25 * W * H):
                    continue
                if not (0.2 <= ar <= 4.0):
                    continue
                
                dets.append(((x, y, w, h), float(sc), int(cid)))
                
        tracks = tracker.step(frame1, frame, dets)
        
        if writer is not None:
            for tid, tr in tracks.items():
                if tr.last_frame != frame1:
                    continue
                x,y,w,h = tr.box
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, f"C{cam_id} L{tid} {tr.score:.2f}", (x, max(0,y-7)), cv2.FONT_HERSHEY_PLAIN, 0.6, (255,255,0), 2)
                
            writer.write(frame)
            
    cap.release()
    if writer is not None:
        writer.release()
        
    tracker.finish()
        
    detection: Dict[int, Detections] = {}
    for lid, tr in tracker.gone.items():
        if not tr.frames:
            continue
        detection[lid] = Detections(cam_id=cam_id, local_id=lid, frames=tr.frames, boxes=tr.boxes, emb=tr.emb.copy() if tr.emb is not None else None)
        
    return Result(detections=detection, width=W, height=H, fps=fps)

def stitch_globals(all_detections, scene_mapper, thresh: float):
    node_list = []
    for cam_id, detections in all_detections.items():
        for lid, detection in detections.items():
            node_list.append((cam_id, lid, detection.start, detection.end, detection.emb))
            
    node_list.sort(key=lambda x: (x[2], x[3]))
    
    first = {}
    def find(a):
        first.setdefault(a, a)
        if first[a] != a:
            first[a] = find(first[a])
        return first[a]
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            first[rb] = ra
            
    for i in range(len(node_list)):
        cam_i, lid_i, start_i, end_i, emb_i = node_list[i]
        if emb_i is None:
            continue
        for j in range(i + 1, len(node_list)):
            cam_j, lid_j, start_j, end_j, emb_j = node_list[j]
            if cam_i == cam_j or emb_j is None or scene_mapper.get(cam_i) != scene_mapper.get(cam_j):
                continue
            
            if cosine_sim(emb_i, emb_j) >= thresh:
                union((cam_i, lid_i), (cam_j, lid_j))
                
    gid_vect, cluster_vect, next_gid = {}, {}, 1
    for cam_id, detections in all_detections.items():
        for lid in detections.keys():
            r = find((cam_id, lid))
            if r not in cluster_vect:
                cluster_vect[r] = next_gid
                next_gid += 1
            gid_vect[(cam_id, lid)] = cluster_vect[r]
    return gid_vect

def prediction(out_dir: str, all_detections: Dict[int, Dict[int, Detections]], gid_vect: Dict[Tuple[int,int], int]):
    with open(out_dir, "w", newline="") as f:
        w = csv.writer(f)
        for cam_id in sorted(all_detections.keys()):
            for lid, seg in all_detections[cam_id].items():
                gid = gid_vect[(cam_id, lid)]
                for fr, (x, y, bw, bh) in zip(seg.frames, seg.boxes):
                    w.writerow([cam_id, gid, fr, x, y, bw, bh, -1, -1])
                    
def group_by_dstype(cams):
    group_list = defaultdict(list)
    for cam_id, vid_path, meta in cams:
        dstype = meta.get("dstype")
        group_list[dstype].append((cam_id, vid_path, meta))
    return group_list

def load_framenum(framenum_file: str) -> dict[int, int]:
    out = {}
    with open(framenum_file) as f:
        for rw in f:
            s = rw.strip()
            parts = re.split(r"[\s]+", s)
            cam_str = parts[0].lower()
            out[int(cam_str[1:])] = int(float(parts[1]))
    return out

def load_offsets(scene_file: str) -> dict[int, float]:
    off = {}
    with open(scene_file) as f:
        for rw in f:
            s = rw.strip()
            parts = re.split(r"[\s]+", s)
            cam_str = parts[0].lower()
            cam_id = int(cam_str[1:])
            off[cam_id] = float(parts[1])
    return off

def cam_time(frame_id: int, fps: float, cam_id: int, scene_id: str, scene_offset: dict[str, dict[int,float]]) -> float | None:
    scene_mapper = scene_offset.get(scene_id, {})
    return scene_mapper[cam_id] + (frame_id - 1) / float(fps)

def load_calibration(calib_path: str) -> Tuple[np.ndarray, float, Optional[np.ndarray], Optional[np.ndarray]]:
    H = None
    err = None
    I = None
    dist = None
    with open(calib_path) as f:
        for rw in f:
            s = rw.strip()
            if s.lower().startswith("homography"):
                hmatrix_str = s.split(":", 1)[1].strip()
                rows = [r.strip() for r in hmatrix_str.split(";") if r.strip()]
                data = []
                for r in rows:
                    nums = [float(x) for x in re.split(r"[,\s]+", r) if x]
                    data.append(nums)
                H = np.array(data, dtype=np.float64).reshape(3,3)
            elif s.lower().startswith("reprojection"):
                val = s.split(":",1)[1].strip()
                err = float(val)
            elif s.lower().startswith("intrinsic"):
                imatrix_str = s.split(":", 1)[1].strip()
                rows = [r.strip() for r in imatrix_str.split(";") if r.strip()]
                data = []
                for r in rows:
                    nums = [float(x) for x in re.split(r"[,\s]+", r) if x]
                    data.append(nums)
                if not data:
                    I = None
                else: I = np.array(data, dtype=np.float64).reshape(3,3)
            elif s.lower().startswith("distortion"):
                dist_str = s.split(":", 1)[1].strip()
                nums = [float(x) for x in re.split(r"[\s]+", dist_str.strip()) if x]
                if not nums:
                    dist = None
                else: dist = np.array(nums, dtype=np.float64)
                                
    return H, err, I, dist

def undistort_world(x: float, y: float, I: Optional[np.ndarray], dist: Optional[np.ndarray]) -> Tuple[float, float]:
    if I is None or dist is None:
        return x, y
    
    coords = np.array([[[x, y]]], dtype=np.float64)
    fixed = cv2.undistortPoints(coords, I, dist, P=I)
    return float(fixed[0,0,0]), float(fixed[0,0,1])

def assign_world(H: np.ndarray, x: float, y: float, I: Optional[np.ndarray] = None, dist: Optional[np.ndarray] = None) -> Tuple[float,float]:
    xu, yu = undistort_world(x, y, I, dist)
    v = np.array([xu, yu, 1.0], dtype=np.float64)
    w = H @ v
    if abs(w[2]) < 1e-9:
        return (-1.0, -1.0)
    return (float(w[0]/w[2]), float(w[1]/w[2]))

def main():
    
    cams = cam_list(cam_list_file)
    cam_groups = group_by_dstype(cams)
    
    #change value to enable drawing (slows execution)
    draw_videos = False
    
    for dstype, cam_group in cam_groups.items():
        prediction_file = os.path.join(out_dir, f"{dstype}_prediction.txt")
        whitener_file = os.path.join(out_dir, f"{dstype}_whitener.npz")
        cam_mapping = {cam_id: {"vdo": vid_path} for cam_id, vid_path, _ in cam_group}
        if dstype != "test":
            dstype_gt_file = os.path.join(eval_dir, f"ground_truth_{dstype}.txt")
            if not os.path.exists(whitener_file):
                X, y = parse_gt(cam_mapping, dstype_gt_file, max_per_id=40)
                if X.size:
                    mu, P = whitener(X, out_dim=128)
                    np.savez(whitener_file, mu=mu, P=P)
            load_whitener(whitener_file)
        else:
            load_whitener(os.path.join(out_dir, "validation_whitener.npz"))
            
            #need to reload since the above creates the file; test uses out of loop instance
            load_whitener(whitener_file)       
    
        all_detections: Dict[int, Dict[int, Detections]] = {}
        
        scene_mapping: Dict[int, str] = {cam_id: meta.get("scene") for cam_id, _, meta in cam_group}
        offsets_cache: Dict[str, Dict[int, float]] = {}
        frame_nums_cache: Dict[str, Dict[int, int]] = {}

        fps_dict: Dict[int, float] = {}
        
        cam_calib = {}
        for cam_id, _, meta in cam_group:
            H, err, I, dist = load_calibration(meta.get("calibration"))
            cam_calib[cam_id] = (H, err, I, dist)
            scene = meta["scene"]
            m = re.search(r"\d+", str(scene))
            scene_stem = f"S{int(m.group(0)):02d}"
            if scene_stem not in offsets_cache:
                offsets_cache[scene_stem] = load_offsets(os.path.join(timestamp_dir, f"{scene_stem}.txt"))
            if scene_stem not in frame_nums_cache:
                frame_nums_cache[scene_stem] = load_framenum(os.path.join(framenum_dir, f"{scene_stem}.txt"))
        
        for cam_id, vid_path, meta in cam_group:
            draw_path = os.path.join(out_dir, f"{dstype}_cam{cam_id:03d}_detections.mp4") if draw_videos else None
            
            scene = meta["scene"]
            scene_stem = f"S{int(re.search(r'\d+', str(scene)).group(0)):02d}"
           
            offsets = offsets_cache[scene_stem]
            frame_nums  = frame_nums_cache[scene_stem]
            
            roi_mask = cv2.imread(meta["roi"], cv2.IMREAD_GRAYSCALE)
            roi_mask = (roi_mask > 0).astype(np.uint8)
        
            cres = camera(cam_id=cam_id, vid_path=vid_path, model=model, conf=conf_val, imgsz=imgsz_val, iou_val=iou_val, max_miss=max_miss, draw=draw_videos, draw_path=draw_path, roi_mask=roi_mask)
            all_detections[cam_id] = cres.detections
            fps_dict[cam_id] = float(cres.fps) if cres.fps else None
            
            expected_frames = frame_nums.get(meta.get("scene"), {}).get(cam_id)
            if expected_frames and draw_videos is False:
                pass

        gid_vect = stitch_globals(all_detections, scene_mapping, thresh)
    
        prediction(prediction_file, all_detections, gid_vect)   

###END FUNCTIONS

if __name__ == "__main__":
    main()

