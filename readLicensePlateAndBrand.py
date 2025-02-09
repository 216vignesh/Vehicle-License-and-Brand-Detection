import cv2
import json
import subprocess
import os
import numpy as np
from ultralytics import YOLO

def run_alpr_on_image(image_path, country="us", topn=3):
    cmd = [
        "alpr",
        "-j",
        "-n", str(topn),
        "-c", country,
        image_path
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        results = json.loads(output.decode('utf-8'))
        return results
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None

class VehicleTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        
    def update(self, boxes, frame_num):
        if not self.tracks:
            return [self.create_new_track(box) for box in boxes]
        
        matched_ids = []
        for box in boxes:
            best_iou = 0
            best_id = None
            for track_id, track in self.tracks.items():
                iou = self.calculate_iou(box, track['box'])
                if iou > best_iou and iou > 0.5:
                    best_iou = iou
                    best_id = track_id
                    
            if best_id is not None:
                self.tracks[best_id]['box'] = box
                self.tracks[best_id]['last_seen'] = frame_num
                matched_ids.append(best_id)
            else:
                matched_ids.append(self.create_new_track(box))
                
        return matched_ids

    def create_new_track(self, box):
        track_id = self.next_id
        self.tracks[track_id] = {
            'box': box,
            'plate': None,
            'confidence': 0,
            'brand': None,
            'last_seen': 0
        }
        self.next_id += 1
        return track_id

    @staticmethod
    def calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / (area1 + area2 - intersection)

def main(video_path, output_path):
    vehicle_model = YOLO('yolov8n.pt')
    brand_model = YOLO('best.pt')
    tracker = VehicleTracker()
    
    cap = cv2.VideoCapture(video_path)
    out_writer = cv2.VideoWriter(
        output_path, 
        cv2.VideoWriter_fourcc(*'XVID'),
        cap.get(cv2.CAP_PROP_FPS),
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )
    
    frame_num = 0
    results_log = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_num += 1
        print(f"\rProcessing frame {frame_num}", end="")

        # Vehicle detection
        vehicle_results = vehicle_model(frame, verbose=False)[0]
        vehicle_boxes = []
        for r in vehicle_results.boxes:
            if r.cls[0] in [2, 3, 5, 7]: 
                box = r.xyxy[0].cpu().numpy()
                vehicle_boxes.append(box)

       
        vehicle_ids = tracker.update(vehicle_boxes, frame_num)

        # Brand detection
        for vid, box in zip(vehicle_ids, vehicle_boxes):
            x1, y1, x2, y2 = box.astype(int)
            sub_img = frame[y1:y2, x1:x2]
            if sub_img.size > 0:
                brand_results = brand_model(sub_img, verbose=False)[0]
                if len(brand_results.boxes) > 0:
                    best_brand_conf = 0
                    best_brand_name = None
                    for b in brand_results.boxes:
                        conf_b = float(b.conf[0])
                        cls_index = int(b.cls[0])
                        brand_name = brand_results.names[cls_index]
                        if conf_b > best_brand_conf:
                            best_brand_conf = conf_b
                            best_brand_name = brand_name
                    if best_brand_conf >= 0.5:
                        tracker.tracks[vid]['brand'] = best_brand_name

        
        temp_path = f"temp_frame_{frame_num}.jpg"
        cv2.imwrite(temp_path, frame)
        alpr_result = run_alpr_on_image(temp_path)
        
        if alpr_result and "results" in alpr_result:
            for plate_obj in alpr_result["results"]:
                coords = plate_obj.get("coordinates", [])
                if len(coords) >= 4:
                    xs = [p["x"] for p in coords]
                    ys = [p["y"] for p in coords]
                    x1p, y1p = min(xs), min(ys)
                    x2p, y2p = max(xs), max(ys)
                    cv2.rectangle(frame, (x1p, y1p), (x2p, y2p), (0,255,0), 2)
                    
                    plate_coords = np.array([[p['x'], p['y']] for p in coords])
                    plate_center = plate_coords.mean(axis=0)
                    
                    
                    best_vehicle_id = None
                    min_dist = float('inf')
                    for vid, box in zip(vehicle_ids, vehicle_boxes):
                        center = [(box[0] + box[2])/2, (box[1] + box[3])/2]
                        dist = np.linalg.norm(plate_center - np.array(center))
                        if dist < min_dist:
                            min_dist = dist
                            best_vehicle_id = vid

                    if best_vehicle_id is not None:
                        current_conf = tracker.tracks[best_vehicle_id]['confidence']
                        plate_conf = float(plate_obj['confidence'])
                        if plate_conf > current_conf:
                            tracker.tracks[best_vehicle_id]['plate'] = plate_obj['plate']
                            tracker.tracks[best_vehicle_id]['confidence'] = plate_conf

        
        for vid, box in zip(vehicle_ids, vehicle_boxes):
            x1v, y1v, x2v, y2v = box.astype(int)
            cv2.rectangle(frame, (x1v, y1v), (x2v, y2v), (255,0,0), 2)
            track_data = tracker.tracks[vid]
            label = f"Vehicle {vid}"
            if track_data['plate']:
                label += f" | {track_data['plate']}"
            if track_data.get('brand'):
                label += f" | {track_data['brand']}"
            cv2.putText(frame, label, (x1v, y1v-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        out_writer.write(frame)
        os.remove(temp_path)

        
        for vid in vehicle_ids:
            trk = tracker.tracks[vid]
            if trk['plate']:
                results_log.append({
                    'frame': frame_num,
                    'vehicle_id': vid,
                    'plate': trk['plate'],
                    'confidence': trk['confidence'],
                    'brand': trk.get('brand')
                })

    print("\nProcessing complete")
    with open('tracking_results.json', 'w') as f:
        json.dump(results_log, f, indent=4)
    
    cap.release()
    out_writer.release()

if __name__ == "__main__":
    main("Segmentation-and-OCR-Test.mp4", "tracked_output.avi")