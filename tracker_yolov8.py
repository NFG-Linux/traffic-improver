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
thresh = 0.7
max_miss = 15
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
            feat = embed_hsv(patch)
            if feat is not None:
                if self.emb is None:
                    self.emb = feat
                else:
                    self.emb = (1 - alpha) * self.emb + alpha * feat
                    
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
            del self.tracks[tid]
            
        return self.tracks
    
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
        cam_dir = os.path.abspath(os.path.normpath(cam_dir))
        #cam_dir = os.path.abspath(cam_dir)
        
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

def crop(frame: np.ndarray, box: Tuple[int,int,int,int]) -> np.ndarray:
    h, w = frame.shape[:2]
    x, y, bw, bh = box
    x2, y2 = x + bw, y + bh
    x, y = max(0, x), max(0, y)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x or y2 <= y:
        return None
    return frame[y:y2, x:x2]

def embed_hsv(img: np.ndarray) -> Optional[np.ndarray]:
    if img is None or img.size == 0:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_bins, s_bins, v_bins = 16, 16, 8
    vect = cv2.calcHist([hsv], [0,1,2], None, [h_bins, s_bins, v_bins], [0,180, 0,256, 0,256])
    vect = cv2.normalize(vect, None).flatten().astype(np.float32)
    return vect

def camera(cam_id: int, vid_path: str, model: YOLO, conf: float, imgsz: int, iou_val: float, max_miss: int, draw: bool,
           draw_path: Optional[str]) -> Result:
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
                x = int(x1); y = int(y1); w = int(x2 - x1); h = int(y2 - y1)
                x, y, w, h = box_frame_bind(x, y, w, h, W, H)
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
        
    detection: Dict[int, Detections] = {}
    for lid, tr in tracker.tracks.items():
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
                    
def gt_compiler(cam_list_by_dstype, dstype_gt_file: str):
     with open(dstype_gt_file, "w", newline="") as out_f:
        w = csv.writer(out_f)
        for cam_id, vid_path, meta in cam_list_by_dstype:
            check = meta.get("dstype")
            if check != "test":
                gt_path = meta.get("gt")
            else: continue
            with open(gt_path) as g:
                for rw in g:
                    s = rw.strip()
                    parts = re.split(r"[,]+", s)
                    fid = int(float(parts[0]))
                    gid = int(float(parts[1]))
                    x   = int(float(parts[2]))
                    y   = int(float(parts[3]))
                    w_  = int(float(parts[4]))
                    h_  = int(float(parts[5]))
                    w.writerow([cam_id, gid, fid, x, y, w_, h_, -1, -1])
                        
def group_by_dstype(cams):
    group_list = defaultdict(list)
    for cam_id, vid_path, meta in cams:
        dstype = meta.get("dstype")
        group_list[dstype].append((cam_id, vid_path, meta))
    return group_list
                    
def main():
    
    cams = cam_list(cam_list_file)
    cam_groups = group_by_dstype(cams)
    
    #change value to enable drawing (slows execution)
    draw_videos = False
    
    for dstype, cam_group in cam_groups.items():
        prediction_file = os.path.join(out_dir, f"{dstype}_prediction.txt")
        dstype_gt_file = os.path.join(out_dir, f"{dstype}_gt.txt")   
    
        all_detections: Dict[int, Dict[int, Detections]] = {}
        scene_mapping = {cam_id: meta.get("scene") for cam_id, _, meta in cam_group}
        
        for cam_id, vid_path, meta in cam_group:
            draw_path = os.path.join(out_dir, f"{dstype}_cam{cam_id:03d}_detections.mp4") if draw_videos else None
        
            cres = camera(cam_id=cam_id, vid_path=vid_path, model=model, conf=conf_val, imgsz=imgsz_val, iou_val=iou_val, max_miss=max_miss, draw=draw_videos, draw_path=draw_path)
            all_detections[cam_id] = cres.detections

        gid_vect = stitch_globals(all_detections, scene_mapping, thresh)
    
        #path = os.path.join(out_dir, prediction_file)
        prediction(prediction_file, all_detections, gid_vect)
    
        if dstype != "test":
            gt_compiler(cam_group, dstype_gt_file)    

###END FUNCTIONS

if __name__ == "__main__":
    main()

