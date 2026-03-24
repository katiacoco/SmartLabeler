# 文件名: canvas.py
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsPolygonItem, QGraphicsRectItem
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt6.QtGui import QPixmap, QPen, QColor, QPolygonF, QPainter
import copy
import math


class ImageCanvas(QGraphicsView):
    click_signal = pyqtSignal(float, float, int)
    box_signal = pyqtSignal(float, float, float, float)
    manual_edit_signal = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self.active_polygon_items = []
        self.confirmed_polygon_items = []

        self.vertex_handles = []
        self.handle_to_vertex = {}
        self.dragging_handle = None
        self.active_polygons_data = []

        self.drawing_box = False
        self.box_start_pos = None
        self.current_box_item = None

    def load_image(self, image_path):
        self._clear_items(self.active_polygon_items)
        self._clear_items(self.confirmed_polygon_items)
        self._clear_items(self.vertex_handles)
        self.handle_to_vertex.clear()

        if self.current_box_item:
            self.scene.removeItem(self.current_box_item)
            self.current_box_item = None

        pixmap = QPixmap(image_path)
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self.resetTransform()
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            zoom_factor = 1.15
        else:
            zoom_factor = 1 / 1.15
        self.scale(zoom_factor, zoom_factor)

    # ==========================================
    # [手术刀 1] 计算点到线段的距离，用于插入新顶点
    # ==========================================
    def point_to_segment_dist(self, px, py, x1, y1, x2, y2):
        l2 = (x1 - x2) ** 2 + (y1 - y2) ** 2
        if l2 == 0: return math.hypot(px - x1, py - y1), x1, y1
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2))
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        return math.hypot(px - proj_x, py - proj_y), proj_x, proj_y

    # ==========================================
    # [手术刀 2] 双击绿线边缘：动态添加红点
    # ==========================================
    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and not self.drawing_box:
            scene_pos = self.mapToScene(event.pos())
            px, py = scene_pos.x(), scene_pos.y()

            min_dist = float('inf')
            best_insert = None

            # 遍历寻找离鼠标双击位置最近的线段
            for poly_idx, poly in enumerate(self.active_polygons_data):
                n = len(poly)
                for i in range(n):
                    p1 = poly[i]
                    p2 = poly[(i + 1) % n]
                    dist, proj_x, proj_y = self.point_to_segment_dist(px, py, p1[0], p1[1], p2[0], p2[1])
                    # 如果距离小于 10 个像素，认为用户点击了该线段
                    if dist < 10 and dist < min_dist:
                        min_dist = dist
                        best_insert = (poly_idx, i + 1, proj_x, proj_y)

            if best_insert:
                poly_idx, insert_idx, ix, iy = best_insert
                self.active_polygons_data[poly_idx].insert(insert_idx, [ix, iy])
                self.draw_active_mask(self.active_polygons_data)
                self.manual_edit_signal.emit(self.active_polygons_data)
                return

        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        if event.modifiers() == Qt.KeyboardModifier.ShiftModifier and event.button() == Qt.MouseButton.LeftButton:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.drawing_box = True
            self.box_start_pos = self.mapToScene(event.pos())
            self.current_box_item = QGraphicsRectItem()
            self.current_box_item.setPen(QPen(QColor(255, 0, 0), 2, Qt.PenStyle.DashLine))
            self.scene.addItem(self.current_box_item)

        elif event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            scene_pos = self.mapToScene(event.pos())
            if self.pixmap_item.isUnderMouse():
                click_type = 1 if event.button() == Qt.MouseButton.LeftButton else 0
                self.click_signal.emit(scene_pos.x(), scene_pos.y(), click_type)

        # ==========================================
        # [手术刀 3] 右键点击红点：直接删除该顶点
        # ==========================================
        elif event.modifiers() == Qt.KeyboardModifier.NoModifier and event.button() == Qt.MouseButton.RightButton:
            scene_pos = self.mapToScene(event.pos())
            item = self.scene.itemAt(scene_pos, self.transform())
            if item in self.handle_to_vertex:
                poly_idx, pt_idx = self.handle_to_vertex[item]
                # 安全锁：多边形最少需要 3 个点
                if len(self.active_polygons_data[poly_idx]) > 3:
                    self.active_polygons_data[poly_idx].pop(pt_idx)
                    self.draw_active_mask(self.active_polygons_data)
                    self.manual_edit_signal.emit(self.active_polygons_data)
                return

        elif event.modifiers() == Qt.KeyboardModifier.NoModifier and event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            item = self.scene.itemAt(scene_pos, self.transform())
            if item in self.handle_to_vertex:
                self.dragging_handle = item
                self.setDragMode(QGraphicsView.DragMode.NoDrag)
                return
            super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing_box and self.box_start_pos:
            current_pos = self.mapToScene(event.pos())
            rect = QRectF(self.box_start_pos, current_pos).normalized()
            self.current_box_item.setRect(rect)

        elif self.dragging_handle:
            scene_pos = self.mapToScene(event.pos())
            self.dragging_handle.setRect(scene_pos.x() - 3, scene_pos.y() - 3, 6, 6)
            poly_idx, pt_idx = self.handle_to_vertex[self.dragging_handle]
            self.active_polygons_data[poly_idx][pt_idx] = [scene_pos.x(), scene_pos.y()]

            poly_item = self.active_polygon_items[poly_idx]
            poly = poly_item.polygon()
            poly[pt_idx] = QPointF(scene_pos.x(), scene_pos.y())
            poly_item.setPolygon(poly)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drawing_box and event.button() == Qt.MouseButton.LeftButton:
            self.drawing_box = False
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            if self.current_box_item:
                rect = self.current_box_item.rect()
                self.scene.removeItem(self.current_box_item)
                self.current_box_item = None
                if rect.width() > 5 and rect.height() > 5:
                    self.box_signal.emit(rect.left(), rect.top(), rect.right(), rect.bottom())

        elif self.dragging_handle and event.button() == Qt.MouseButton.LeftButton:
            self.dragging_handle = None
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.manual_edit_signal.emit(self.active_polygons_data)
        else:
            super().mouseReleaseEvent(event)

    def _clear_items(self, item_list):
        for item in item_list:
            if item.scene() == self.scene:
                self.scene.removeItem(item)
        item_list.clear()

    def draw_active_mask(self, polygons):
        self.clear_active_mask()
        self.active_polygons_data = copy.deepcopy(polygons)

        for poly_idx, poly_points in enumerate(self.active_polygons_data):
            if len(poly_points) < 3: continue
            q_points = [QPointF(float(pt[0]), float(pt[1])) for pt in poly_points]
            polygon_item = QGraphicsPolygonItem(QPolygonF(q_points))
            polygon_item.setBrush(QColor(0, 255, 0, 120))
            polygon_item.setPen(QPen(QColor(0, 255, 0), 2))
            self.scene.addItem(polygon_item)
            self.active_polygon_items.append(polygon_item)

            for pt_idx, pt in enumerate(poly_points):
                handle = QGraphicsRectItem(float(pt[0]) - 3, float(pt[1]) - 3, 6, 6)
                handle.setBrush(QColor(255, 0, 0))
                handle.setPen(QPen(Qt.GlobalColor.white, 1))
                handle.setZValue(10)
                self.scene.addItem(handle)
                self.vertex_handles.append(handle)
                self.handle_to_vertex[handle] = (poly_idx, pt_idx)

    def commit_to_confirmed(self, class_name):
        self.clear_active_mask()

    def clear_active_mask(self):
        self._clear_items(self.active_polygon_items)
        self._clear_items(self.vertex_handles)
        self.handle_to_vertex.clear()
        self.active_polygons_data = []

    def redraw_all_confirmed(self, annotations, hidden_indices=set(), alpha=80):
        self._clear_items(self.confirmed_polygon_items)
        for idx, ann in enumerate(annotations):
            if idx in hidden_indices:
                continue

            for poly_points in ann['polygons']:
                if len(poly_points) < 3: continue
                q_points = [QPointF(float(pt[0]), float(pt[1])) for pt in poly_points]
                polygon_item = QGraphicsPolygonItem(QPolygonF(q_points))

                polygon_item.setBrush(QColor(0, 100, 255, alpha))
                polygon_item.setPen(QPen(QColor(0, 100, 255), 1 if alpha > 0 else 0))

                self.scene.addItem(polygon_item)
                self.confirmed_polygon_items.append(polygon_item)