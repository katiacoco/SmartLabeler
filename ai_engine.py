# 文件名: ai_engine.py
import torch
import cv2
import numpy as np
from ultralytics import SAM, YOLO


class SAMModel:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"正在加载 SAM 抠图大模型 (设备: {self.device})...")
        self.model = SAM('mobile_sam.pt')

        print(f"正在加载 YOLOv8 语义侦察雷达 (设备: {self.device})...")
        self.detector = YOLO('yolov8n.pt')

        self.current_image = None

    def set_image(self, image_path):
        self.current_image = image_path

    def auto_detect_boxes(self, target_class_dict, conf_thresh=0.5):
        if not self.current_image: return []

        results = self.detector(self.current_image, verbose=False)
        if not results or len(results) == 0: return []

        result = results[0]
        boxes = []

        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < conf_thresh: continue

            class_id = int(box.cls[0])
            yolo_class_name = result.names[class_id].lower()

            for t_class, t_id in target_class_dict.items():
                if t_class.lower() in yolo_class_name or yolo_class_name in t_class.lower():
                    boxes.append({
                        'coords': box.xyxy[0].tolist(),
                        'class_id': t_id,
                        'class_name': t_class
                    })
                    break

        return boxes

    def predict(self, points=None, labels=None, bboxes=None):
        if not self.current_image: return []

        inference_args = {
            'source': self.current_image,
            'device': self.device,
            'retina_masks': True,
            'imgsz': 1024,
            'verbose': False
        }

        if points and labels and len(points) > 0:
            inference_args['points'] = [points]
            inference_args['labels'] = [labels]

        if bboxes and len(bboxes) > 0:
            inference_args['bboxes'] = [bboxes[-1]]

        if 'points' not in inference_args and 'bboxes' not in inference_args:
            return []

        results = self.model(**inference_args)

        if results and results[0].masks:
            raw_polygons = results[0].masks.xy
            clean_polygons = []

            for poly in raw_polygons:
                if len(poly) > 15:
                    # ==========================================
                    # [极度安全的内存隔离]
                    # 强制转为 float32 且使用 .copy() 重新申请独立内存！
                    # 彻底阻断 C++ 底层越界崩溃 (0xC0000409)
                    # ==========================================
                    poly_np = np.array(poly, dtype=np.float32).copy()

                    perimeter = cv2.arcLength(poly_np, True)
                    epsilon = 0.002 * perimeter
                    approx_poly = cv2.approxPolyDP(poly_np, epsilon, closed=True)
                    approx_poly = approx_poly.reshape(-1, 2)

                    if len(approx_poly) >= 3:
                        # 必须使用 .tolist() 阻断内存共享
                        clean_polygons.append(approx_poly.tolist())

            return clean_polygons
        return []