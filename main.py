# 文件名: main.py
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import json
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
import csv
import shutil
import random

from PyQt6.QtWidgets import (QApplication, QMainWindow, QHBoxLayout, QVBoxLayout,
                             QWidget, QPushButton, QFileDialog, QLabel, QComboBox,
                             QListWidget, QMessageBox, QInputDialog, QDialog,
                             QTextEdit, QGroupBox, QProgressDialog, QListWidgetItem, QSlider, QProgressBar)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QMutex, QWaitCondition
from PyQt6.QtGui import QColor, QPalette

from ai_engine import SAMModel
from canvas import ImageCanvas


class SplashLoader(QThread):
    loaded = pyqtSignal(object)

    def run(self):
        ai = SAMModel()
        self.loaded.emit(ai)


# ==========================================
# [新增] 赛博朋克数据统计仪表盘
# ==========================================
class AnalyticsDashboard(QDialog):
    def __init__(self, counts, class_list, total_files, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📊 数据集全局罗盘 (Data Analytics)")
        self.resize(650, 500)
        self.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")

        layout = QVBoxLayout()

        # 头部概览
        header = QLabel("AI 训练样本分布状态")
        header.setStyleSheet("font-size: 22px; font-weight: bold; color: #00e5ff; margin-bottom: 5px;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        total_objects = sum(counts.values())
        sub_header = QLabel(f"已扫描 {total_files} 张图片 | 共捕获目标: {total_objects} 个")
        sub_header.setStyleSheet("font-size: 14px; color: #b0bec5; margin-bottom: 20px;")
        sub_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(sub_header)

        # 进度条列表区
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        for cid in range(len(class_list)):
            count = counts.get(cid, 0)
            cname = class_list[cid]
            pct = (count / total_objects * 100) if total_objects > 0 else 0

            row_layout = QHBoxLayout()

            # 标签
            lbl_name = QLabel(f"[{cid}] {cname}: {count} 个 ({pct:.1f}%)")
            lbl_name.setFixedWidth(200)

            # 进度条
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(pct))
            bar.setTextVisible(False)
            bar.setFixedHeight(15)

            # 智能颜色诊断
            if pct == 0:
                color = "#424242"  # 灰色 (毫无数据)
                lbl_name.setStyleSheet("color: #757575;")
            elif pct < 5:
                color = "#ff1744"  # 红色预警 (极度缺乏样本)
                lbl_name.setStyleSheet("color: #ff1744; font-weight: bold;")
                lbl_name.setText(lbl_name.text() + " ⚠️极度缺乏")
            elif pct < 15:
                color = "#ffb300"  # 黄色警告
                lbl_name.setStyleSheet("color: #ffb300;")
            else:
                color = "#00e676"  # 绿色健康
                lbl_name.setStyleSheet("color: #ffffff;")

            bar.setStyleSheet(f"""
                QProgressBar {{ border: 1px solid #424242; border-radius: 5px; background: #2c2c2c; }}
                QProgressBar::chunk {{ background-color: {color}; border-radius: 5px; }}
            """)

            row_layout.addWidget(lbl_name)
            row_layout.addWidget(bar)
            scroll_layout.addLayout(row_layout)

        scroll_layout.addStretch()
        layout.addWidget(scroll_widget)

        btn_close = QPushButton("关闭面板")
        btn_close.setStyleSheet("background-color: #37474f; font-weight: bold; height: 35px; border-radius: 5px;")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

        self.setLayout(layout)


# ==========================================
# [新增] YOLO 标准训练包后台生成线程
# ==========================================
class YoloExportThread(QThread):
    progress_signal = pyqtSignal(int, int, str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, image_files, txt_dir, output_dir, train_ratio, class_list):
        super().__init__()
        self.image_files = image_files
        self.txt_dir = txt_dir
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.class_list = class_list

    def run(self):
        try:
            # 1. 过滤出有 txt 标注的图片
            valid_pairs = []
            for img_path in self.image_files:
                base = os.path.splitext(os.path.basename(img_path))[0]
                txt_path = os.path.join(self.txt_dir, base + ".txt")
                if os.path.exists(txt_path):
                    valid_pairs.append((img_path, txt_path))

            if not valid_pairs:
                self.error_signal.emit("未找到任何已标注的数据，请先进行标注！")
                return

            # 2. 随机打乱并计算分割比例
            random.shuffle(valid_pairs)
            split_idx = int(len(valid_pairs) * (self.train_ratio / 100.0))
            train_pairs = valid_pairs[:split_idx]
            val_pairs = valid_pairs[split_idx:]

            # 3. 创建 YOLO 标准文件夹结构
            dirs = {
                'img_tr': os.path.join(self.output_dir, 'images', 'train'),
                'img_val': os.path.join(self.output_dir, 'images', 'val'),
                'lbl_tr': os.path.join(self.output_dir, 'labels', 'train'),
                'lbl_val': os.path.join(self.output_dir, 'labels', 'val')
            }
            for d in dirs.values():
                os.makedirs(d, exist_ok=True)

            # 4. 拷贝文件
            total = len(valid_pairs)
            current = 0
            for pair_list, img_d, lbl_d in [(train_pairs, dirs['img_tr'], dirs['lbl_tr']),
                                            (val_pairs, dirs['img_val'], dirs['lbl_val'])]:
                for img_p, txt_p in pair_list:
                    current += 1
                    self.progress_signal.emit(current, total, os.path.basename(img_p))
                    shutil.copy(img_p, os.path.join(img_d, os.path.basename(img_p)))
                    shutil.copy(txt_p, os.path.join(lbl_d, os.path.basename(txt_p)))

            # 5. 自动生成 data.yaml
            yaml_path = os.path.join(self.output_dir, 'data.yaml')
            with open(yaml_path, 'w', encoding='utf-8') as f:
                f.write("path: ./  # dataset root dir\n")
                f.write("train: images/train  # train images (relative to 'path')\n")
                f.write("val: images/val  # val images (relative to 'path')\n\n")
                f.write(f"nc: {len(self.class_list)}  # number of classes\n")
                f.write("names:\n")
                for i, c in enumerate(self.class_list):
                    f.write(f"  {i}: '{c}'\n")

            self.finished_signal.emit(
                f"成功导出 {len(train_pairs)} 张训练集，{len(val_pairs)} 张验证集。\ndata.yaml 已完美生成！")

        except Exception as e:
            self.error_signal.emit(str(e))


class BatchConfigDialog(QDialog):
    def __init__(self, image_paths, class_list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("多类别混合批量生产线")
        self.resize(600, 580)
        self.image_paths = image_paths

        main_layout = QVBoxLayout()
        h_layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel(f"共发现 {len(image_paths)} 张图片："))
        self.img_list_widget = QListWidget()
        for path in image_paths:
            item = QListWidgetItem(os.path.basename(path))
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.img_list_widget.addItem(item)
        left_layout.addWidget(self.img_list_widget)
        h_layout.addLayout(left_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("选择要让 AI 同时寻找的类别："))
        self.cls_list_widget = QListWidget()
        for class_name in class_list:
            item = QListWidgetItem(class_name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.cls_list_widget.addItem(item)

        if self.cls_list_widget.count() > 0:
            self.cls_list_widget.item(0).setCheckState(Qt.CheckState.Checked)

        right_layout.addWidget(self.cls_list_widget)
        h_layout.addLayout(right_layout)

        main_layout.addLayout(h_layout)

        format_layout = QVBoxLayout()
        format_layout.addWidget(QLabel("选择批量导出的数据格式："))
        self.cb_format = QComboBox()
        self.cb_format.addItems([
            "YOLO 分割格式 (*.txt) [每图独立]",
            "YOLO 目标检测格式 (*.txt) [每图独立]",
            "PASCAL VOC 格式 (*.xml) [每图独立]",
            "COCO JSON 格式 (*.json) [全局合并单文件]",
            "CSV 表格 (*.csv) [全局合并单文件]"
        ])
        self.cb_format.setStyleSheet("height: 35px; font-weight: bold; background-color: #f5f5f5; border-radius: 4px;")
        format_layout.addWidget(self.cb_format)
        main_layout.addLayout(format_layout)
        main_layout.addSpacing(10)

        action_layout = QHBoxLayout()
        btn_ok = QPushButton("🚀 开始混合批量处理")
        btn_ok.setStyleSheet("background-color: #e8f5e9; font-weight: bold; height: 35px;")
        btn_cancel = QPushButton("取消")
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
        action_layout.addWidget(btn_cancel)
        action_layout.addWidget(btn_ok)
        main_layout.addLayout(action_layout)

        self.setLayout(main_layout)

    def get_selected_paths(self):
        return [self.img_list_widget.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.img_list_widget.count()) if
                self.img_list_widget.item(i).checkState() == Qt.CheckState.Checked]

    def get_target_class_dict(self, full_class_list):
        t_dict = {}
        for i in range(self.cls_list_widget.count()):
            item = self.cls_list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                name = item.text()
                t_dict[name] = full_class_list.index(name)
        return t_dict

    def get_export_format(self):
        return self.cb_format.currentText()


class AIEngineWorker(QThread):
    model_loaded = pyqtSignal()
    manual_result = pyqtSignal(list)
    auto_result = pyqtSignal(list)
    batch_progress = pyqtSignal(int, int, str)
    batch_finished = pyqtSignal()
    error_signal = pyqtSignal(str)
    cancelled_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.ai = None
        self.queue = []
        self.mutex = QMutex()
        self.cond = QWaitCondition()
        self.is_running = True
        self.is_cancelled = False

    def add_task(self, task):
        self.mutex.lock()
        if task['type'] == 'manual':
            self.queue = [t for t in self.queue if t['type'] != 'manual']
        self.queue.append(task)
        self.cond.wakeOne()
        self.mutex.unlock()

    def cancel_task(self):
        self.is_cancelled = True

    def stop(self):
        self.is_running = False
        self.cond.wakeOne()
        self.wait()

    def run(self):
        try:
            self.ai = SAMModel()
            self.model_loaded.emit()
        except Exception as e:
            self.error_signal.emit(f"模型加载失败: {e}")
            return

        while self.is_running:
            self.mutex.lock()
            if not self.queue:
                self.cond.wait(self.mutex)
            if not self.is_running:
                self.mutex.unlock()
                break
            task = self.queue.pop(0)
            self.mutex.unlock()

            self.is_cancelled = False
            try:
                task_type = task['type']

                if task_type == 'manual':
                    self.ai.set_image(task['image_path'])
                    polys = self.ai.predict(points=task['points'], labels=task['labels'], bboxes=task['bboxes'])
                    if not self.is_cancelled:
                        self.manual_result.emit(polys)

                elif task_type == 'auto':
                    self.ai.set_image(task['image_path'])
                    candidate_boxes = self.ai.auto_detect_boxes(task['target_class_dict'], task['conf_thresh'])
                    all_polys = []
                    if candidate_boxes:
                        for item in candidate_boxes:
                            if self.is_cancelled: break
                            p = self.ai.predict(bboxes=[item['coords']])
                            if p: all_polys.extend(p)

                    if self.is_cancelled:
                        self.cancelled_signal.emit()
                    else:
                        self.auto_result.emit(all_polys)

                elif task_type == 'batch':
                    img_paths = task['image_paths']
                    t_dict = task['target_class_dict']
                    c_thresh = task['conf_thresh']
                    s_dir = task['save_dir']
                    exp_fmt = task['export_format']
                    cls_list = task['class_list']

                    coco_data = {"images": [], "categories": [{"id": i, "name": n} for i, n in enumerate(cls_list)],
                                 "annotations": []}
                    csv_data = []
                    ann_id = 0

                    total = len(img_paths)
                    for i, img_path in enumerate(img_paths):
                        if self.is_cancelled: break
                        base_name = os.path.basename(img_path)
                        self.batch_progress.emit(i + 1, total, base_name)

                        self.ai.set_image(img_path)
                        img = cv2.imread(img_path)
                        if img is None: continue
                        img_h, img_w = img.shape[:2]

                        candidate_boxes = self.ai.auto_detect_boxes(t_dict, c_thresh)
                        if not candidate_boxes: continue

                        class_polys = {}
                        for item in candidate_boxes:
                            if self.is_cancelled: break
                            poly = self.ai.predict(bboxes=[item['coords']])
                            if poly:
                                cid = item['class_id']
                                if cid not in class_polys: class_polys[cid] = []
                                class_polys[cid].extend(poly)

                        if self.is_cancelled: break
                        if not class_polys: continue

                        save_d = s_dir if s_dir else os.path.dirname(img_path)
                        os.makedirs(save_d, exist_ok=True)
                        base_no_ext = os.path.splitext(base_name)[0]

                        if "YOLO 分割" in exp_fmt:
                            txt_path = os.path.join(save_d, base_no_ext + ".txt")
                            existing_lines = []
                            if os.path.exists(txt_path):
                                with open(txt_path, 'r', encoding='utf-8') as f:
                                    existing_lines = [line.strip() for line in f.readlines() if line.strip()]
                            new_lines = []
                            for cid, polys in class_polys.items():
                                for poly in polys:
                                    if len(poly) < 3: continue
                                    parts = [str(cid)]
                                    for pt in poly:
                                        parts.extend([f"{max(0., min(1., float(pt[0]) / img_w)):.6f}",
                                                      f"{max(0., min(1., float(pt[1]) / img_h)):.6f}"])
                                    new_lines.append(" ".join(parts))
                            unique_lines = list(dict.fromkeys(existing_lines + new_lines))
                            with open(txt_path, 'w', encoding='utf-8') as f:
                                for line in unique_lines: f.write(line + "\n")

                        elif "YOLO 目标检测" in exp_fmt:
                            txt_path = os.path.join(save_d, base_no_ext + ".txt")
                            existing_lines = []
                            if os.path.exists(txt_path):
                                with open(txt_path, 'r', encoding='utf-8') as f:
                                    existing_lines = [line.strip() for line in f.readlines() if line.strip()]
                            new_lines = []
                            for cid, polys in class_polys.items():
                                for poly in polys:
                                    if len(poly) < 3: continue
                                    ax = [pt[0] for pt in poly];
                                    ay = [pt[1] for pt in poly]
                                    cx, cy = ((min(ax) + max(ax)) / 2) / img_w, ((min(ay) + max(ay)) / 2) / img_h
                                    w, h = (max(ax) - min(ax)) / img_w, (max(ay) - min(ay)) / img_h
                                    new_lines.append(
                                        f"{cid} {max(0, min(1, cx)):.6f} {max(0, min(1, cy)):.6f} {max(0, min(1, w)):.6f} {max(0, min(1, h)):.6f}")
                            unique_lines = list(dict.fromkeys(existing_lines + new_lines))
                            with open(txt_path, 'w', encoding='utf-8') as f:
                                for line in unique_lines: f.write(line + "\n")

                        elif "VOC" in exp_fmt:
                            xml_path = os.path.join(save_d, base_no_ext + ".xml")
                            a = ET.Element("annotation");
                            ET.SubElement(a, "filename").text = base_name
                            sz = ET.SubElement(a, "size");
                            ET.SubElement(sz, "width").text = str(int(img_w));
                            ET.SubElement(sz, "height").text = str(int(img_h))
                            for cid, polys in class_polys.items():
                                cname = cls_list[cid]
                                for poly in polys:
                                    if len(poly) < 3: continue
                                    ax = [pt[0] for pt in poly];
                                    ay = [pt[1] for pt in poly]
                                    ob = ET.SubElement(a, "object");
                                    ET.SubElement(ob, "name").text = cname
                                    b = ET.SubElement(ob, "bndbox")
                                    ET.SubElement(b, "xmin").text = str(int(min(ax)));
                                    ET.SubElement(b, "ymin").text = str(int(min(ay)))
                                    ET.SubElement(b, "xmax").text = str(int(max(ax)));
                                    ET.SubElement(b, "ymax").text = str(int(max(ay)))
                            with open(xml_path, 'w', encoding='utf-8') as f:
                                f.write(minidom.parseString(ET.tostring(a)).toprettyxml(indent="  "))

                        elif "COCO" in exp_fmt:
                            coco_data["images"].append(
                                {"id": i, "file_name": base_name, "width": int(img_w), "height": int(img_h)})
                            for cid, polys in class_polys.items():
                                for poly in polys:
                                    fp = []
                                    for pt in poly: fp.extend([pt[0], pt[1]])
                                    if not fp: continue
                                    xs, ys = fp[0::2], fp[1::2]
                                    w, h = max(xs) - min(xs), max(ys) - min(ys)
                                    coco_data["annotations"].append({
                                        "id": ann_id, "image_id": i, "category_id": cid, "segmentation": [fp],
                                        "area": w * h, "bbox": [min(xs), min(ys), w, h], "iscrowd": 0
                                    })
                                    ann_id += 1

                        elif "CSV" in exp_fmt:
                            for cid, polys in class_polys.items():
                                for poly in polys:
                                    poly_str = ";".join([f"{pt[0]:.2f};{pt[1]:.2f}" for pt in poly])
                                    csv_data.append([base_name, int(img_w), int(img_h), cid, cls_list[cid], poly_str])

                    if not self.is_cancelled:
                        if "COCO" in exp_fmt and coco_data["images"]:
                            with open(os.path.join(s_dir, "batch_coco_annotations.json"), 'w', encoding='utf-8') as f:
                                json.dump(coco_data, f, indent=4)
                        elif "CSV" in exp_fmt and csv_data:
                            with open(os.path.join(s_dir, "batch_annotations.csv"), 'w', newline='',
                                      encoding='utf-8') as f:
                                w = csv.writer(f)
                                w.writerow(
                                    ['image_name', 'width', 'height', 'class_id', 'class_name', 'polygon_points'])
                                w.writerows(csv_data)

                        self.batch_finished.emit()

            except Exception as e:
                self.error_signal.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 智能标注助手")
        self.resize(1300, 850)

        self.canvas = ImageCanvas()

        self.image_files = []
        self.current_img_idx = -1
        self.current_image_path = ""
        self.current_image_size = (0, 0)
        self.config_file = "smartlabeler_config.json"
        self.last_save_dir = ""
        self._load_config()

        self.active_points, self.active_labels, self.active_bboxes, self.active_polygons = [], [], [], []
        self.prompt_history, self.annotations = [], []
        self.class_list = ["car", "person", "dog", "cat", "bus", "truck", "motorcycle", "bicycle"]

        self._is_loading_list = False
        self._is_inferencing = False

        self._setup_ui()
        self.canvas.click_signal.connect(self.handle_point_inference)
        self.canvas.box_signal.connect(self.handle_box_inference)
        self.canvas.manual_edit_signal.connect(self.sync_manual_edit)

        self.splash = QProgressDialog("正在安全环境隔离唤醒 PyTorch 大模型...\n（引擎启动中，请耐心等待，彻底告别闪退）",
                                      None, 0, 0, self)
        self.splash.setWindowTitle("SmartLabeler 极速启动中")
        self.splash.setWindowModality(Qt.WindowModality.WindowModal)
        self.splash.setCancelButton(None)
        self.splash.show()
        QApplication.processEvents()

        self.ai_worker = AIEngineWorker()
        self.ai_worker.model_loaded.connect(self.on_model_loaded)
        self.ai_worker.manual_result.connect(self.on_inference_success)
        self.ai_worker.auto_result.connect(self.on_auto_label_success)
        self.ai_worker.batch_progress.connect(self.on_batch_progress)
        self.ai_worker.batch_finished.connect(self.on_batch_finished)
        self.ai_worker.error_signal.connect(self.on_inference_error)
        self.ai_worker.cancelled_signal.connect(self.on_ai_cancelled)
        self.ai_worker.start()

    def closeEvent(self, event):
        if hasattr(self, 'ai_worker'):
            self.ai_worker.stop()
        event.accept()

    def on_model_loaded(self):
        self.splash.close()
        self.setEnabled(True)
        self.lbl_status.setText("状态: 双击绿线加点 | 右击红点删点 | Space 确认|shift+左键拖拽 精准选中|更多功能请细看说明书")

    def _load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.last_save_dir = json.load(f).get("last_save_dir", "")
            except:
                pass

    def _save_config(self):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump({"last_save_dir": self.last_save_dir}, f, ensure_ascii=False, indent=2)
        except:
            pass

    def keyPressEvent(self, event):
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_Z:
            self.undo_action()
        elif event.key() == Qt.Key.Key_Space:
            self.commit_object()
        elif event.key() == Qt.Key.Key_A:
            self.prev_image()
        elif event.key() == Qt.Key.Key_D:
            self.next_image()
        elif Qt.Key.Key_1 <= event.key() <= Qt.Key.Key_9:
            idx = event.key() - Qt.Key.Key_1
            if idx < self.cb_classes.count(): self.cb_classes.setCurrentIndex(idx)
        else:
            super().keyPressEvent(event)

    def _setup_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        self.lbl_status = QLabel("状态: 正在唤醒底层神经网络，请稍候...")
        self.lbl_status.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")
        left_layout.addWidget(self.lbl_status)
        left_layout.addWidget(self.canvas)

        right_panel = QWidget()
        right_panel.setFixedWidth(340)
        right_layout = QVBoxLayout()

        # --- 顶部高级工具 ---
        tools_group = QGroupBox("💼 数据工程师高级面板")
        tools_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #4fc3f7; border-radius: 5px; margin-top: 10px; padding-top: 15px; }")
        tools_layout = QVBoxLayout()

        self.btn_analytics = QPushButton("📊 赛博数据全局统计大屏")
        self.btn_analytics.setStyleSheet(
            "background-color: #212121; color: #00e5ff; font-weight: bold; height: 35px; border: 1px solid #00e5ff;")
        self.btn_analytics.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # <--- 【补回免死金牌】禁止抢占焦点
        self.btn_analytics.clicked.connect(self.show_analytics_dashboard)
        tools_layout.addWidget(self.btn_analytics)

        self.btn_export_yolo = QPushButton("📦 一键生成 YOLO 标准训练包")
        self.btn_export_yolo.setStyleSheet(
            "background-color: #212121; color: #ffab00; font-weight: bold; height: 35px; border: 1px solid #ffab00;")
        self.btn_export_yolo.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # <--- 【补回免死金牌】禁止抢占焦点
        self.btn_export_yolo.clicked.connect(self.trigger_yolo_export)
        tools_layout.addWidget(self.btn_export_yolo)
        tools_group.setLayout(tools_layout)

        # --- 基础工具 ---
        file_ctrl_layout = QHBoxLayout()
        self.btn_open_file = QPushButton("单张打开");
        self.btn_open_file.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_open_folder = QPushButton("📁 批量打开");
        self.btn_open_folder.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_open_file.clicked.connect(self.open_image);
        self.btn_open_folder.clicked.connect(self.open_folder)
        file_ctrl_layout.addWidget(self.btn_open_file);
        file_ctrl_layout.addWidget(self.btn_open_folder)

        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◀ 上图 (A)");
        self.btn_prev.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_next = QPushButton("下图 (D) ▶");
        self.btn_next.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_prev.clicked.connect(self.prev_image);
        self.btn_next.clicked.connect(self.next_image)
        nav_layout.addWidget(self.btn_prev);
        nav_layout.addWidget(self.btn_next)

        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("👁️ 透明度:"))
        self.slider_alpha = QSlider(Qt.Orientation.Horizontal)
        self.slider_alpha.setRange(0, 255)
        self.slider_alpha.setValue(80)
        self.slider_alpha.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.slider_alpha.valueChanged.connect(self.update_canvas_render)
        alpha_layout.addWidget(self.slider_alpha)

        class_layout = QHBoxLayout()
        self.cb_classes = QComboBox();
        self.cb_classes.addItems(self.class_list);
        self.cb_classes.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_add_class = QPushButton("➕ 新增");
        self.btn_add_class.setFixedWidth(60);
        self.btn_add_class.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_add_class.clicked.connect(self.add_custom_class)
        class_layout.addWidget(self.cb_classes);
        class_layout.addWidget(self.btn_add_class)

        ai_group = QGroupBox("🚀 零样本 AI 侦察引擎")
        ai_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid silver; border-radius: 5px; margin-top: 10px; padding-top: 15px; }")
        ai_layout = QVBoxLayout()

        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("🎯 AI灵敏度:"))
        self.slider_conf = QSlider(Qt.Orientation.Horizontal)
        self.slider_conf.setRange(10, 90)
        self.slider_conf.setValue(50)
        self.slider_conf.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.lbl_conf_val = QLabel("50%")
        self.slider_conf.valueChanged.connect(lambda v: self.lbl_conf_val.setText(f"{v}%"))
        conf_layout.addWidget(self.slider_conf)
        conf_layout.addWidget(self.lbl_conf_val)
        ai_layout.addLayout(conf_layout)

        self.btn_auto_label = QPushButton("🤖 扫描当前图片")
        self.btn_auto_label.setMinimumHeight(40)
        self.btn_auto_label.setStyleSheet("background-color: #ede7f6; font-weight: bold; color: #4527a0;")
        self.btn_auto_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_auto_label.clicked.connect(self.trigger_ai_auto_label)
        ai_layout.addWidget(self.btn_auto_label)

        self.btn_batch_label = QPushButton("🗂️ 多类别混合批量标注...")
        self.btn_batch_label.setMinimumHeight(40)
        self.btn_batch_label.setStyleSheet("background-color: #fff3e0; font-weight: bold; color: #e65100;")
        self.btn_batch_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_batch_label.clicked.connect(self.trigger_batch_auto_label)
        ai_layout.addWidget(self.btn_batch_label)

        ai_group.setLayout(ai_layout)

        self.btn_commit = QPushButton("2. 确认当前草稿 (Space)")
        self.btn_commit.setMinimumHeight(40)
        self.btn_commit.setStyleSheet("background-color: #e0f7fa; font-weight: bold;")
        self.btn_commit.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_commit.clicked.connect(self.commit_object)

        self.list_annotations = QListWidget()
        self.list_annotations.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.list_annotations.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.list_annotations.itemChanged.connect(self.on_layer_visibility_changed)

        list_ctrl_layout = QHBoxLayout()
        self.btn_edit = QPushButton("✏️ 修改选中");
        self.btn_edit.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_del = QPushButton("🗑️ 删除选中");
        self.btn_del.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_edit.clicked.connect(self.edit_selected_object);
        self.btn_del.clicked.connect(self.delete_selected_object)
        list_ctrl_layout.addWidget(self.btn_edit);
        list_ctrl_layout.addWidget(self.btn_del)

        self.btn_help = QPushButton("📖 软件说明书 (Help)")
        self.btn_help.setStyleSheet(
            "background-color: #fff9c4; font-weight: bold; color: #f57f17; border: 2px solid #fbc02d; border-radius: 6px;")
        self.btn_help.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_help.clicked.connect(self.show_help_manual)

        right_layout.addWidget(tools_group)
        right_layout.addLayout(file_ctrl_layout)
        right_layout.addLayout(nav_layout)
        right_layout.addLayout(alpha_layout)
        right_layout.addWidget(QLabel("当前类别 [按 1-9 切换]:"))
        right_layout.addLayout(class_layout)
        right_layout.addWidget(ai_group)
        right_layout.addWidget(self.btn_commit)
        right_layout.addWidget(QLabel("已确认图层:"))
        right_layout.addWidget(self.list_annotations)
        right_layout.addLayout(list_ctrl_layout)
        right_layout.addStretch()
        right_layout.addWidget(self.btn_help)

        right_panel.setLayout(right_layout)
        main_layout.addLayout(left_layout)
        main_layout.addWidget(right_panel)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        self.setEnabled(False)

    # ==========================================
    # [分析 1] 数据聚合逻辑
    # ==========================================
    def show_analytics_dashboard(self):
        target_dir = self.last_save_dir
        if not target_dir or not os.path.exists(target_dir):
            target_dir = QFileDialog.getExistingDirectory(self, "请选择您的标注存放文件夹(如 training_txt)")
            if not target_dir: return

        counts = {}
        total_files = 0
        for f in os.listdir(target_dir):
            if f.endswith('.txt') and f != 'classes.txt':
                total_files += 1
                try:
                    with open(os.path.join(target_dir, f), 'r') as txt:
                        for line in txt:
                            parts = line.strip().split()
                            if parts:
                                cid = int(parts[0])
                                counts[cid] = counts.get(cid, 0) + 1
                except:
                    pass

        dialog = AnalyticsDashboard(counts, self.class_list, total_files, self)
        dialog.exec()

    # ==========================================
    # [打包 1] YOLO 标准训练包构建流
    # ==========================================
    def trigger_yolo_export(self):
        if not self.image_files:
            QMessageBox.warning(self, "警告", "请先打开包含图片的文件夹！")
            return
        if not self.last_save_dir:
            QMessageBox.warning(self, "警告", "请至少产生过一次标注记录，再进行打包。")
            return

        ratio, ok = QInputDialog.getInt(self, "数据拆分", "请输入训练集所占比例 (例如输入 80 代表 8:2 拆分):", 80, 10,
                                        99)
        if not ok: return

        out_dir = QFileDialog.getExistingDirectory(self, "选择存放 YOLO 标准训练包的根目录")
        if not out_dir: return

        self.export_thread = YoloExportThread(self.image_files, self.last_save_dir, out_dir, ratio, self.class_list)

        self.progress_dialog = QProgressDialog("正在构建 YOLO 标准数据目录...", "🛑 取消", 0, len(self.image_files),
                                               self)
        self.progress_dialog.setWindowTitle("打包中")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.show()

        self.export_thread.progress_signal.connect(lambda c, t, n: self.progress_dialog.setValue(c))
        self.export_thread.finished_signal.connect(self.on_export_finished)
        self.export_thread.error_signal.connect(lambda e: QMessageBox.critical(self, "错误", e))
        self.export_thread.start()

    def on_export_finished(self, msg):
        if hasattr(self, 'progress_dialog'): self.progress_dialog.close()
        QMessageBox.information(self, "打包完成", msg)

    def update_canvas_render(self):
        if self._is_loading_list: return
        hidden_indices = set()
        for i in range(self.list_annotations.count()):
            if self.list_annotations.item(i).checkState() == Qt.CheckState.Unchecked:
                hidden_indices.add(i)
        self.canvas.redraw_all_confirmed(self.annotations, hidden_indices, self.slider_alpha.value())

    def on_layer_visibility_changed(self, item):
        self.update_canvas_render()

    def _sync_list_widget(self):
        self._is_loading_list = True
        self.list_annotations.clear()
        for ann in self.annotations:
            item = QListWidgetItem(f"[{ann['class_id']}] {ann['class_name']} (Polys: {len(ann['polygons'])})")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.list_annotations.addItem(item)
        self._is_loading_list = False
        self.update_canvas_render()

    def trigger_batch_auto_label(self):
        if not self.image_files:
            QMessageBox.warning(self, "提示", "请先点击【📁 批量打开】选择图片文件夹。")
            return

        dialog = BatchConfigDialog(self.image_files, self.class_list, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_paths = dialog.get_selected_paths()
            target_class_dict = dialog.get_target_class_dict(self.class_list)
            export_format = dialog.get_export_format()

            if not selected_paths or not target_class_dict:
                return

            default_dir = self.last_save_dir if self.last_save_dir else os.path.dirname(self.image_files[0])
            save_dir = QFileDialog.getExistingDirectory(self, "请选择批量导出的存放文件夹", default_dir)
            if not save_dir: return

            self.last_save_dir = save_dir
            self._save_config()

            self.canvas.setEnabled(True)
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

            self.progress_dialog = QProgressDialog("AI 正在多线程批量爆破中...", "🛑 取消", 0, len(selected_paths), self)
            self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            self.progress_dialog.canceled.connect(self.ai_worker.cancel_task)

            self.ai_worker.add_task({
                'type': 'batch',
                'image_paths': selected_paths,
                'target_class_dict': target_class_dict,
                'conf_thresh': self.slider_conf.value() / 100.0,
                'save_dir': save_dir,
                'export_format': export_format,
                'class_list': self.class_list
            })

    def on_batch_progress(self, current, total, filename):
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.setLabelText(f"扫描: {filename} ({current}/{total})")
            self.progress_dialog.setValue(current)

    def on_batch_finished(self):
        self._restore_ui_state()
        QMessageBox.information(self, "批量完成", "🎉 批量任务完美收工！")
        if self.current_image_path:
            self._load_image_at_index(self.current_img_idx)

    def trigger_ai_auto_label(self):
        if not self.current_image_path: return
        self.canvas.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.lbl_status.setText("状态：YOLO & SAM 联合扫描中...")
        self.lbl_status.setStyleSheet("color: blue; font-weight: bold;")

        self.progress_dialog = QProgressDialog(f"扫描单图...", "🛑 取消", 0, 0, self)
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.canceled.connect(self.ai_worker.cancel_task)
        self.progress_dialog.show()

        self.ai_worker.add_task({
            'type': 'auto',
            'image_path': self.current_image_path,
            'target_class_dict': {self.cb_classes.currentText(): self.cb_classes.currentIndex()},
            'conf_thresh': self.slider_conf.value() / 100.0
        })

    def on_ai_cancelled(self):
        self._restore_ui_state()
        self.lbl_status.setText("状态：已取消 AI 任务。")

    def on_auto_label_success(self, polygons):
        self.active_polygons = polygons
        if len(self.active_polygons) > 0:
            self.canvas.draw_active_mask(self.active_polygons)
        self._restore_ui_state()

    def prev_image(self):
        self._auto_save_current_annotations()
        if self.current_img_idx > 0:
            self.current_img_idx -= 1
            self._load_image_at_index(self.current_img_idx)

    def next_image(self):
        self._auto_save_current_annotations()
        if self.current_img_idx < len(self.image_files) - 1:
            self.current_img_idx += 1
            self._load_image_at_index(self.current_img_idx)

    def _auto_save_current_annotations(self):
        if not self.current_image_path: return
        base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
        save_dir = self.last_save_dir if getattr(self, 'last_save_dir', '') else os.path.dirname(
            self.current_image_path)
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
            except:
                pass
        save_path = os.path.join(save_dir, base_name + ".txt")
        img_w, img_h = self.current_image_size
        if img_w == 0 or img_h == 0: return

        if not self.annotations:
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                except:
                    pass
            return

        try:
            unique_lines = []
            seen_lines = set()
            for ann in self.annotations:
                for poly in ann['polygons']:
                    if len(poly) < 3: continue
                    parts = [str(ann['class_id'])]
                    for pt in poly:
                        parts.extend([f"{max(0., min(1., float(pt[0]) / img_w)):.6f}",
                                      f"{max(0., min(1., float(pt[1]) / img_h)):.6f}"])
                    line_str = " ".join(parts) + "\n"
                    if line_str not in seen_lines:
                        seen_lines.add(line_str)
                        unique_lines.append(line_str)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.writelines(unique_lines)
        except:
            pass

    def _reverse_load_annotations(self):
        base_name = os.path.splitext(os.path.basename(self.current_image_path))[0] + ".txt"
        path_1 = os.path.join(os.path.dirname(self.current_image_path), base_name)
        path_2 = os.path.join(self.last_save_dir, base_name) if getattr(self, 'last_save_dir', '') else ""

        path_3 = ""
        parent_dir = os.path.dirname(os.path.dirname(self.current_image_path))
        possible_label_dirs = ['training_txt', 'labels', 'Annotations']
        for d in possible_label_dirs:
            test_path = os.path.join(parent_dir, d, base_name)
            if os.path.exists(test_path):
                path_3 = test_path
                break

        txt_path = ""
        if os.path.exists(path_1):
            txt_path = path_1
        elif path_2 and os.path.exists(path_2):
            txt_path = path_2
        elif path_3 and os.path.exists(path_3):
            txt_path = path_3

        if not txt_path or not os.path.exists(txt_path): return

        img_w, img_h = self.current_image_size
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            unique_lines = list(dict.fromkeys([line.strip() for line in lines if line.strip()]))
            for line in unique_lines:
                parts = line.split()
                if len(parts) < 7: continue
                cid = int(parts[0])
                cname = self.class_list[cid] if cid < len(self.class_list) else f"cls_{cid}"
                poly = [[float(parts[i]) * img_w, float(parts[i + 1]) * img_h] for i in range(1, len(parts), 2)]
                self.annotations.append({'class_id': cid, 'class_name': cname, 'polygons': [poly]})
            self._sync_list_widget()
        except:
            pass

    def _restore_ui_state(self):
        self._is_inferencing = False
        self.canvas.setEnabled(True)
        while QApplication.overrideCursor() is not None:
            QApplication.restoreOverrideCursor()
        self.lbl_status.setText("状态: 双击绿线加点 | 右击红点删点 | Space 确认")
        self.lbl_status.setStyleSheet("color: green; font-weight: bold;")
        if hasattr(self, 'progress_dialog') and self.progress_dialog is not None:
            self.progress_dialog.close()
            self.progress_dialog = None

    def on_inference_success(self, polygons):
        self.active_polygons = polygons
        if len(self.active_polygons) > 0: self.canvas.draw_active_mask(self.active_polygons)
        self._restore_ui_state()

    def on_inference_error(self, error_msg):
        self._restore_ui_state()
        QMessageBox.warning(self, "错误", f"AI 推理错误: {error_msg}")

    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择图片")
        if not folder_path: return
        self.image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                                   f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        if self.image_files:
            self.current_img_idx = 0
            self._load_image_at_index(self.current_img_idx)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择单张图片", "", "Images (*.png *.jpg *.jpeg)")
        if path: self.image_files = [path]; self.current_img_idx = 0; self._load_image_at_index(0)

    def _load_image_at_index(self, index):
        try:
            path = self.image_files[index]
            self.current_image_path = path
            self.canvas.load_image(path)
            self.current_image_size = (
            self.canvas.pixmap_item.pixmap().width(), self.canvas.pixmap_item.pixmap().height())
            self.annotations.clear()
            self._sync_list_widget()
            self.clear_active_state()
            self.setWindowTitle(f"AI 智能标注 - {os.path.basename(path)} [{index + 1}/{len(self.image_files)}]")
            self._reverse_load_annotations()
        except:
            pass

    def commit_object(self):
        if not self.active_polygons: return
        cidx = self.cb_classes.currentIndex()
        cname = self.cb_classes.currentText()
        self.annotations.append({'class_id': cidx, 'class_name': cname, 'polygons': self.active_polygons.copy()})
        self._sync_list_widget()
        self.canvas.commit_to_confirmed(cname)
        self.clear_active_state()
        self._auto_save_current_annotations()

    def edit_selected_object(self):
        sel = self.list_annotations.selectedItems()
        if not sel: return
        row = self.list_annotations.row(sel[0])
        if self.active_polygons:
            if QMessageBox.question(self, "确认", "丢弃草稿加载修改？") == QMessageBox.StandardButton.No: return
        ann = self.annotations.pop(row)
        self._sync_list_widget()
        self.clear_active_state()
        self.active_polygons = ann['polygons']
        self.cb_classes.setCurrentText(ann['class_name'])
        self.canvas.draw_active_mask(self.active_polygons)
        self.update_canvas_render()
        self._auto_save_current_annotations()

    def delete_selected_object(self):
        sel = self.list_annotations.selectedItems()
        if not sel: return
        row = self.list_annotations.row(sel[0])
        self.annotations.pop(row)
        self._sync_list_widget()
        self.update_canvas_render()
        self._auto_save_current_annotations()

    def undo_action(self):
        if self._is_inferencing: return
        if not self.current_image_path: return
        if self.prompt_history:
            la = self.prompt_history.pop()
            if la == 'point':
                self.active_points.pop();
                self.active_labels.pop()
            elif la == 'box':
                self.active_bboxes.pop()
            if self.prompt_history:
                self._trigger_inference()
            else:
                self.active_polygons = [];
                self.canvas.clear_active_mask()
            return
        if self.annotations:
            self.annotations.pop()
            self._sync_list_widget()
            self.update_canvas_render()
            self._auto_save_current_annotations()

    def sync_manual_edit(self, updated_polygons):
        self.active_polygons = updated_polygons

    def add_custom_class(self):
        nc, ok = QInputDialog.getText(self, "新增", "输入新标签:")
        if ok and nc.strip():
            nc = nc.strip()
            if nc not in self.class_list:
                self.class_list.append(nc);
                self.cb_classes.addItem(nc)
            self.cb_classes.setCurrentText(nc)

    def handle_point_inference(self, x, y, label):
        if not self.current_image_path: return
        if self._is_inferencing: return

        img_w, img_h = self.current_image_size
        safe_x = max(0.0, min(float(x), float(img_w)))
        safe_y = max(0.0, min(float(y), float(img_h)))

        self.active_points.append([safe_x, safe_y])
        self.active_labels.append(label)
        self.prompt_history.append('point')
        self._trigger_inference()

    def handle_box_inference(self, x_min, y_min, x_max, y_max):
        if not self.current_image_path: return
        if self._is_inferencing: return

        img_w, img_h = self.current_image_size
        safe_xmin = max(0.0, min(float(x_min), float(img_w)))
        safe_ymin = max(0.0, min(float(y_min), float(img_h)))
        safe_xmax = max(0.0, min(float(x_max), float(img_w)))
        safe_ymax = max(0.0, min(float(y_max), float(img_h)))

        if safe_xmax - safe_xmin < 1 or safe_ymax - safe_ymin < 1: return

        self.active_bboxes.append([safe_xmin, safe_ymin, safe_xmax, safe_ymax])
        self.prompt_history.append('box')
        self._trigger_inference()

    def _trigger_inference(self):
        self._is_inferencing = True
        self.canvas.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.lbl_status.setText("状态：AI 推理中，请稍候...")
        self.lbl_status.setStyleSheet("color: red; font-weight: bold;")
        QApplication.processEvents()

        self.ai_worker.add_task({
            'type': 'manual',
            'image_path': self.current_image_path,
            'points': self.active_points,
            'labels': self.active_labels,
            'bboxes': self.active_bboxes
        })

    def clear_active_state(self):
        self.active_points.clear();
        self.active_labels.clear();
        self.active_bboxes.clear();
        self.prompt_history.clear()
        self.active_polygons = [];
        self.canvas.clear_active_mask()

    def show_help_manual(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("SmartLabeler 使用说明书")
        dialog.resize(650, 700)
        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet("font-size: 14px; line-height: 1.6; font-family: 'Microsoft YaHei', Consolas;")
        try:
            with open("README.txt", "r", encoding="utf-8") as f:
                text_edit.setText(f.read())
        except:
            text_edit.setText("⚠️ 未找到说明书文件！请确保名为 'README.txt' 的文件在同一目录下。")
        layout.addWidget(text_edit)
        btn_close = QPushButton("我已了解 (Close)")
        btn_close.setMinimumHeight(40)
        btn_close.clicked.connect(dialog.accept)
        layout.addWidget(btn_close)
        dialog.setLayout(layout)
        dialog.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())