"""
CADIX - Detector Principal
Detector mejorado con mejor manejo de errores y performance.
"""

import time
import threading
import numpy as np
import cv2 as cv
import onnxruntime as ort
from typing import Tuple, List, Optional, Callable
from datetime import datetime

from src.core.logger import get_logger
from src.config.settings import SystemConfig

class CADIXDetector(threading.Thread):
    """Detector principal mejorado para uso industrial"""
    
    def __init__(self, 
                 config: SystemConfig,
                 frame_callback: Callable,
                 stats_callback: Callable,
                 status_callback: Callable,
                 shot_callback: Optional[Callable] = None,
                 session_id: str = None):
        
        super().__init__(daemon=True)
        self.config = config
        self.frame_cb = frame_callback
        self.stats_cb = stats_callback
        self.status_cb = status_callback
        self.shot_cb = shot_callback
        self.session_id = session_id
        
        self.logger = get_logger()
        self.stop_flag = threading.Event()
        
        # Variables de estado
        self.fps = 0.0
        self.ema_val = None
        self.hist = []
        self.lock_bbox = None
        self.lock_bbox_smooth = None
        self.lock_misses = 0
        self.prev_center = None
        self.refractory = 0
        self.last_beep_t = 0.0
        
        # Métricas de performance
        self.frame_count = 0
        self.detection_count = 0
        self.alert_count = 0
        self.avg_confidence = 0.0
        
        # Contadores FPS
        self._fps_n = 0
        self._fps_t0 = time.time()
        self._plot_tick = 0
        
        # Inicializar detector
        self._init_onnx_session()
        self._init_camera()
        
        self.logger.info("Detector inicializado correctamente")
    
    def _init_onnx_session(self):
        """Inicializa la sesión ONNX"""
        try:
            providers = ort.get_available_providers()
            if "CPUExecutionProvider" not in providers:
                providers.append("CPUExecutionProvider")
            
            self.sess = ort.InferenceSession(self.config.detection.model_path, providers=providers)
            self.inp = self.sess.get_inputs()[0]
            
            # Determinar layout del modelo
            shp = [int(d) if isinstance(d, (int, np.integer)) else -1 for d in self.inp.shape]
            self.layout = "NCHW" if (len(shp) == 4 and (shp[1] in (1, 3))) else "NHWC"
            
            self.logger.info(f"Modelo ONNX cargado: {self.config.detection.model_path}")
            self.logger.info(f"Providers disponibles: {providers}")
            
        except Exception as e:
            self.logger.error(f"Error cargando modelo ONNX: {e}")
            raise
    
    def _init_camera(self):
        """Inicializa la cámara"""
        try:
            self.cap = self._find_working_webcam()
            if self.cap is None:
                raise RuntimeError("No se pudo inicializar la cámara")
            
            self.logger.log_camera_event(
                "Cámara inicializada",
                f"Índice: {self.config.camera.index} | Resolución: {self.config.camera.width}x{self.config.camera.height}"
            )
            
        except Exception as e:
            self.logger.error(f"Error inicializando cámara: {e}")
            raise
    
    def _find_working_webcam(self):
        """Encuentra una cámara funcional"""
        def try_camera(index):
            cap = cv.VideoCapture(index, cv.CAP_MSMF)
            if not cap.isOpened():
                return None
            
            # Configurar cámara
            cap.set(cv.CAP_PROP_FPS, self.config.camera.fps)
            cap.set(cv.CAP_PROP_FRAME_WIDTH, self.config.camera.width)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.config.camera.height)
            
            try:
                cap.set(cv.CAP_PROP_BUFFERSIZE, self.config.camera.buffer_size)
            except:
                pass
            
            # Test de captura
            for _ in range(3):
                ret, _ = cap.read()
                if ret:
                    return cap
                time.sleep(0.01)
            
            cap.release()
            return None
        
        # Intentar índice preferido primero
        cap = try_camera(self.config.camera.index)
        if cap:
            return cap
        
        # Intentar otros índices
        for idx in range(0, 4):
            if idx != self.config.camera.index:
                cap = try_camera(idx)
                if cap:
                    self.config.camera.index = idx
                    return cap
        
        return None
    
    def _grab_frame(self):
        """Captura un frame de la cámara"""
        try:
            # Vaciar buffer
            for _ in range(1):
                self.cap.grab()
            
            ret, frame = self.cap.read()
            if not ret:
                return None
            
            return cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        except Exception as e:
            self.logger.error(f"Error capturando frame: {e}")
            return None
    
    def _enhance_image(self, bgr):
        """Mejora la imagen si está habilitado"""
        if not self.config.image_processing.low_light_enhance:
            return bgr
        
        try:
            lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
            l, a, b = cv.split(lab)
            
            clahe = cv.createCLAHE(
                clipLimit=self.config.image_processing.clahe_clip_limit,
                tileGridSize=self.config.image_processing.clahe_tile_size
            )
            l2 = clahe.apply(l)
            
            lab2 = cv.merge((l2, a, b))
            enhanced = cv.cvtColor(lab2, cv.COLOR_LAB2BGR)
            
            if self.config.image_processing.denoise_on_dark:
                enhanced = cv.fastNlMeansDenoisingColored(enhanced, None, 3, 3, 7, 21)
            
            return enhanced
        
        except Exception as e:
            self.logger.error(f"Error mejorando imagen: {e}")
            return bgr
    
    def _letterbox(self, im, new_shape=(480, 480), color=(114, 114, 114)):
        """Redimensiona imagen manteniendo aspecto"""
        h, w = im.shape[:2]
        r = min(new_shape[0]/h, new_shape[1]/w)
        nw, nh = int(round(w*r)), int(round(h*r))
        
        im_res = cv.resize(im, (nw, nh), interpolation=cv.INTER_LINEAR)
        dw, dh = (new_shape[1]-nw)/2, (new_shape[0]-nh)/2
        
        top = int(round(dh-0.1))
        bottom = int(round(dh+0.1))
        left = int(round(dw-0.1))
        right = int(round(dw+0.1))
        
        im_pad = cv.copyMakeBorder(im_res, top, bottom, left, right,
                                   cv.BORDER_CONSTANT, value=color)
        
        return im_pad, r, (dw, dh)
    
    def _decode_predictions(self, out0, model_w, model_h):
        """Decodifica predicciones del modelo"""
        try:
            a = out0
            if a.ndim != 3 or a.shape[0] != 1:
                return np.zeros((0, 4)), np.zeros((0,))
            
            if a.shape[1] == 5:
                arr = a[0].T
            elif a.shape[2] == 5:
                arr = a[0]
            else:
                return np.zeros((0, 4)), np.zeros((0,))
            
            if arr.size == 0:
                return np.zeros((0, 4)), np.zeros((0,))
            
            coords = arr[:, :4].astype(np.float32)
            conf = arr[:, 4].astype(np.float32)
            
            # Filtrar por confianza
            mask = conf >= self.config.detection.confidence_threshold
            coords = coords[mask]
            conf = conf[mask]
            
            if len(coords) == 0:
                return np.zeros((0, 4)), np.zeros((0,))
            
            # Convertir de xywh a xyxy si es necesario
            norm = np.max(np.abs(coords)) <= 1.5
            xyxy = coords.copy()
            
            valid_frac = np.mean((xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1]))
            if valid_frac < 0.2:
                # Formato xywh -> xyxy
                cx, cy, w, h = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2
                xyxy = np.stack([x1, y1, x2, y2], 1)
            
            # Desnormalizar si es necesario
            if norm:
                xyxy[:, [0, 2]] *= model_w
                xyxy[:, [1, 3]] *= model_h
            
            return xyxy, conf
        
        except Exception as e:
            self.logger.error(f"Error decodificando predicciones: {e}")
            return np.zeros((0, 4)), np.zeros((0,))
    
    def _apply_nms(self, boxes, scores):
        """Aplica Non-Maximum Suppression"""
        if len(boxes) == 0:
            return np.array([], dtype=np.int32)
        
        def iou_matrix(box, boxes):
            xx1 = np.maximum(box[0], boxes[:, 0])
            yy1 = np.maximum(box[1], boxes[:, 1])
            xx2 = np.minimum(box[2], boxes[:, 2])
            yy2 = np.minimum(box[3], boxes[:, 3])
            
            w = np.maximum(0, xx2-xx1)
            h = np.maximum(0, yy2-yy1)
            inter = w * h
            
            area_b = (box[2]-box[0])*(box[3]-box[1])
            area_bs = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
            
            return inter / (area_b + area_bs - inter + 1e-6)
        
        idxs = scores.argsort()[::-1]
        keep = []
        
        while idxs.size:
            i = idxs[0]
            keep.append(i)
            
            if idxs.size == 1:
                break
            
            ious = iou_matrix(boxes[i], boxes[idxs[1:]])
            idxs = idxs[1:][ious < self.config.detection.iou_threshold]
        
        return np.array(keep, dtype=np.int32)
    
    def _bbox_iou(self, a, b):
        """Calcula IoU entre dos bounding boxes"""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        
        xx1 = max(ax1, bx1)
        yy1 = max(ay1, by1)
        xx2 = min(ax2, bx2)
        yy2 = min(ay2, by2)
        
        w = max(0, xx2-xx1)
        h = max(0, yy2-yy1)
        inter = w * h
        
        area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
        area_b = max(0, bx2-bx1) * max(0, by2-by1)
        
        return inter / (area_a + area_b - inter + 1e-6)
    
    def _track_object(self, boxes, scores):
        """Sistema de tracking de objetos mejorado"""
        if len(boxes) == 0:
            if self.lock_bbox is not None:
                self.lock_misses += 1
                if self.lock_misses > self.config.detection.lock_max_misses:
                    self.lock_bbox = None
                    self.lock_bbox_smooth = None
                    self.lock_misses = 0
                    self.logger.debug("Tracking perdido - reiniciando")
            return None, None
        
        if self.lock_bbox is None:
            # Inicializar tracking con la mejor detección
            best_idx = int(np.argmax(scores))
            chosen = boxes[best_idx].astype(float)
            
            self.lock_bbox = chosen.copy()
            self.lock_bbox_smooth = chosen.copy()
            self.lock_misses = 0
            
            self.logger.debug(f"Tracking iniciado con confianza: {scores[best_idx]:.3f}")
            return chosen, scores[best_idx]
        
        # Encontrar mejor match con objeto trackeado
        best_iou, best_idx = -1.0, -1
        
        for i, box in enumerate(boxes):
            iou = self._bbox_iou(self.lock_bbox, box)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        
        if best_iou >= self.config.detection.lock_iou_threshold:
            # Buen match encontrado
            chosen = boxes[best_idx].astype(float)
            self.lock_bbox = chosen.copy()
            self.lock_misses = 0
            
            # Suavizar bbox para display
            smooth_factor = self.config.detection.lock_smoothing
            self.lock_bbox_smooth = (
                smooth_factor * chosen + 
                (1.0 - smooth_factor) * self.lock_bbox_smooth
            )
            
            return chosen, scores[best_idx]
        
        else:
            # No hay buen match
            self.lock_misses += 1
            
            if self.lock_misses <= self.config.detection.lock_max_misses:
                # Usar último bbox conocido
                return self.lock_bbox.copy(), 0.0
            else:
                # Reiniciar tracking
                best_idx = int(np.argmax(scores))
                chosen = boxes[best_idx].astype(float)
                
                self.lock_bbox = chosen.copy()
                self.lock_bbox_smooth = chosen.copy()
                self.lock_misses = 0
                
                self.logger.debug("Tracking reiniciado")
                return chosen, scores[best_idx]
    
    def _calculate_measurement(self, bbox, W, H):
        """Calcula la medición desde el bounding box"""
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        
        w_box = max(1, x2 - x1 + 1)
        margin = int(round(self.config.measurement.margin_fraction * w_box))
        
        X1 = max(0, x1 + margin)
        X2 = min(W-1, x2 - margin)
        Yc = int(round(0.5 * (y1 + y2)))
        
        px_dist = max(0, X2 - X1)
        mm_dist = px_dist * self.config.measurement.px_to_mm
        
        # Actualizar EMA
        alpha = self.config.measurement.ema_alpha
        self.ema_val = (
            alpha * mm_dist + (1.0 - alpha) * (self.ema_val if self.ema_val is not None else mm_dist)
        )
        
        return mm_dist, (X1, Yc, X2, Yc)
    
    def _check_oneshot(self, bbox, mm_dist, W, H):
        """Verifica condición de one-shot"""
        x1, y1, x2, y2 = bbox
        cx, cy = int(0.5 * (x1 + x2)), int(0.5 * (y1 + y2))
        
        gate_x = int(self.config.oneshot.gate_x_ratio * W)
        gate_y = H // 2
        y_tolerance = int(self.config.oneshot.gate_y_tolerance_fraction * H)
        
        if self.prev_center is None:
            self.prev_center = (cx, cy)
            return
        
        if self.refractory > 0:
            self.refractory -= 1
        
        # Verificar cruce
        prev_side = np.sign(self.prev_center[0] - gate_x)
        curr_side = np.sign(cx - gate_x)
        crossed = (prev_side != 0) and (curr_side != 0) and (prev_side != curr_side)
        
        # Verificar dirección
        shot_dir = self.config.oneshot.shot_direction
        dir_ok = (
            shot_dir == "any" or
            (shot_dir == "L2R" and self.prev_center[0] < gate_x <= cx) or
            (shot_dir == "R2L" and self.prev_center[0] > gate_x >= cx)
        )
        
        # Verificar posición vertical
        near_center_y = abs(cy - gate_y) <= y_tolerance
        
        if crossed and dir_ok and near_center_y and self.refractory == 0:
            # Disparar one-shot
            value = (
                float(self.ema_val) if (self.config.measurement.use_ema_for_shots and self.ema_val is not None)
                else float(mm_dist)
            )
            
            direction = "L2R" if self.prev_center[0] < cx else "R2L"
            
            if self.shot_cb:
                try:
                    self.shot_cb(value, direction, self.session_id)
                except Exception as e:
                    self.logger.error(f"Error en callback de one-shot: {e}")
            
            self.refractory = self.config.oneshot.refractory_frames
            self.logger.info(f"One-shot disparado: {value:.2f}mm ({direction})")
        
        self.prev_center = (cx, cy)
    
    def _check_alerts(self, mm_dist):
        """Verifica condiciones de alerta"""
        if mm_dist > self.config.measurement.alert_threshold_mm:
            now = time.time()
            if now - self.last_beep_t >= self.config.measurement.alert_cooldown_s:
                # Disparar alerta
                self._trigger_alert()
                self.last_beep_t = now
                self.alert_count += 1
    
    def _trigger_alert(self):
        """Dispara una alerta"""
        try:
            import platform
            if platform.system().lower().startswith("win"):
                try:
                    import winsound
                    winsound.Beep(1200, 150)
                except:
                    print('\a', end='', flush=True)
            else:
                print('\a', end='', flush=True)
        except Exception as e:
            self.logger.error(f"Error reproduciendo alerta: {e}")
    
    def _draw_overlays(self, image, boxes, scores, tracked_bbox, measurement_line, W, H):
        """Dibuja overlays en la imagen"""
        try:
            # Línea del gate
            gate_x = int(self.config.oneshot.gate_x_ratio * W)
            cv.line(image, (gate_x, 0), (gate_x, H), (0, 0, 255), 1, cv.LINE_AA)
            
            # Bounding boxes de todas las detecciones
            for (x1, y1, x2, y2), score in zip(boxes.astype(int), scores):
                cv.rectangle(image, (x1, y1), (x2, y2), (80, 180, 230), 1, cv.LINE_AA)
                cv.putText(image, f"{score:.2f}", (x1, y1-5), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.4, (80, 180, 230), 1)
            
            # Bbox trackeado (suavizado)
            if self.lock_bbox_smooth is not None:
                x1, y1, x2, y2 = [int(round(v)) for v in self.lock_bbox_smooth]
                cv.rectangle(image, (x1, y1), (x2, y2), (50, 220, 120), 2, cv.LINE_AA)
            
            # Línea de medición
            if measurement_line:
                x1, y, x2, _ = measurement_line
                cv.line(image, (x1, y), (x2, y), (255, 0, 255), 2, cv.LINE_AA)
                cv.circle(image, (x1, y), 4, (255, 0, 0), -1, cv.LINE_AA)
                cv.circle(image, (x2, y), 4, (255, 0, 0), -1, cv.LINE_AA)
        
        except Exception as e:
            self.logger.error(f"Error dibujando overlays: {e}")
    
    def _draw_info(self, image, mm_dist, has_detection, W, H):
        """Dibuja información en la imagen"""
        try:
            # Estado de detección
            if has_detection:
                status_text = "DETECTADO"
                status_color = (60, 200, 120)
                
                # Información de medición
                text = f"{mm_dist/10.0:.2f} cm  ({mm_dist:.1f} mm)"
                if self.ema_val:
                    text += f"  EMA {self.ema_val:.1f}"
                
                cv.putText(image, text, (20, H-60), cv.FONT_HERSHEY_SIMPLEX, 
                          0.6, (0, 255, 255), 2, cv.LINE_AA)
                
                # Alerta si excede umbral
                if mm_dist > self.config.measurement.alert_threshold_mm:
                    cv.putText(image, "⚠ ALERTA >120mm", (W//2-100, 50), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv.LINE_AA)
            else:
                status_text = "NO DETECTADO"
                status_color = (230, 80, 80)
            
            # Estado
            cv.putText(image, status_text, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 
                      0.8, status_color, 2, cv.LINE_AA)
            
            # FPS
            cv.putText(image, f"FPS: {self.fps:.1f}", (20, 60), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)
            
            # Información de sesión
            if self.session_id:
                cv.putText(image, f"Sesión: {self.session_id}", (20, H-20), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv.LINE_AA)
        
        except Exception as e:
            self.logger.error(f"Error dibujando información: {e}")
    
    def _update_fps(self):
        """Actualiza cálculo de FPS"""
        self._fps_n += 1
        if self._fps_n % 10 == 0:
            t1 = time.time()
            dt = t1 - self._fps_t0
            if dt > 0:
                self.fps = 10.0 / dt
            self._fps_t0 = t1
    
    def _update_stats(self, mm_dist, confidence):
        """Actualiza estadísticas"""
        if mm_dist is not None:
            self.hist.append(float(mm_dist))
            if len(self.hist) > 400:
                self.hist = self.hist[-200:]  # Mantener últimas 200
            
            avg_mm = float(np.mean(self.hist))
            
            # Callback de estadísticas
            plot_update = (self._plot_tick % 8 == 0)  # Actualizar gráfico cada 8 frames
            
            try:
                self.stats_cb(mm_dist, self.ema_val or mm_dist, avg_mm, plot_update)
            except Exception as e:
                self.logger.error(f"Error en callback de estadísticas: {e}")
            
            self._plot_tick += 1
        
        # Actualizar confianza promedio
        if confidence is not None:
            self.avg_confidence = 0.9 * self.avg_confidence + 0.1 * confidence
        
        # Log periódico de estadísticas
        if self.frame_count % 100 == 0:
            self.logger.log_detection_stats(self.fps, self.detection_count, self.avg_confidence)
    
    def run(self):
        """Loop principal del detector"""
        self.status_cb("Detector iniciado")
        self.logger.info("Iniciando loop principal del detector")
        
        try:
            while not self.stop_flag.is_set():
                start_time = time.time()
                
                # Capturar frame
                rgb_frame = self._grab_frame()
                if rgb_frame is None:
                    time.sleep(0.005)
                    continue
                
                self.frame_count += 1
                
                # Convertir y mejorar imagen
                bgr = cv.cvtColor(rgb_frame, cv.COLOR_RGB2BGR)
                bgr = self._enhance_image(bgr)
                
                H, W = bgr.shape[:2]
                
                # Preparar imagen para inferencia
                letterboxed, ratio, (dw, dh) = self._letterbox(
                    bgr, (self.config.detection.input_size, self.config.detection.input_size)
                )
                
                rgb_normalized = cv.cvtColor(letterboxed, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
                
                # Preparar input tensor
                if self.layout == "NCHW":
                    input_tensor = np.transpose(rgb_normalized, (2, 0, 1))[None, ...]
                else:
                    input_tensor = rgb_normalized[None, ...]
                
                # Inferencia
                try:
                    outputs = self.sess.run(None, {self.inp.name: input_tensor})
                    boxes, scores = self._decode_predictions(
                        outputs[0], self.config.detection.input_size, self.config.detection.input_size
                    )
                except Exception as e:
                    self.logger.error(f"Error en inferencia: {e}")
                    continue
                
                # NMS
                if len(boxes) > 0:
                    keep_indices = self._apply_nms(boxes, scores)
                    boxes, scores = boxes[keep_indices], scores[keep_indices]
                
                # Convertir coordenadas de vuelta a imagen original
                if len(boxes) > 0:
                    boxes[:, [0, 2]] -= dw
                    boxes[:, [1, 3]] -= dh
                    boxes /= ratio
                    
                    # Clip a límites de imagen
                    boxes[:, 0] = np.clip(boxes[:, 0], 0, W-1)
                    boxes[:, 2] = np.clip(boxes[:, 2], 0, W-1)
                    boxes[:, 1] = np.clip(boxes[:, 1], 0, H-1)
                    boxes[:, 3] = np.clip(boxes[:, 3], 0, H-1)
                
                # Tracking y medición
                tracked_box, confidence = self._track_object(boxes, scores)
                
                mm_dist = None
                measurement_line = None
                
                if tracked_box is not None:
                    self.detection_count += 1
                    
                    # Calcular medición
                    mm_dist, measurement_line = self._calculate_measurement(tracked_box, W, H)
                    
                    # Verificar one-shot
                    self._check_oneshot(tracked_box, mm_dist, W, H)
                    
                    # Verificar alertas
                    self._check_alerts(mm_dist)
                else:
                    # Reset tracking para one-shot
                    self.prev_center = None
                
                # Dibujar overlays
                output_image = bgr.copy()
                self._draw_overlays(output_image, boxes, scores, tracked_box, measurement_line, W, H)
                self._draw_info(output_image, mm_dist or 0, tracked_box is not None, W, H)
                
                # Callbacks
                try:
                    self.frame_cb(output_image)
                except Exception as e:
                    self.logger.error(f"Error en callback de frame: {e}")
                
                # Actualizar métricas
                self._update_fps()
                self._update_stats(mm_dist, confidence)
                
                # Control de velocidad
                elapsed = time.time() - start_time
                min_frame_time = 1.0 / 60.0  # Máximo 60 FPS
                if elapsed < min_frame_time:
                    time.sleep(min_frame_time - elapsed)
        
        except Exception as e:
            self.logger.error(f"Error en loop principal: {e}")
            self.status_cb(f"Error en detector: {e}")
        
        finally:
            self.cleanup()
            self.status_cb("Detector detenido")
            self.logger.info("Detector finalizado")
    
    def stop(self):
        """Detiene el detector"""
        self.stop_flag.set()
    
    def cleanup(self):
        """Limpia recursos"""
        try:
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
                self.logger.log_camera_event("Cámara liberada")
        except Exception as e:
            self.logger.error(f"Error liberando recursos: {e}")
    
    def get_stats(self) -> dict:
        """Obtiene estadísticas del detector"""
        return {
            'fps': self.fps,
            'frame_count': self.frame_count,
            'detection_count': self.detection_count,
            'alert_count': self.alert_count,
            'avg_confidence': self.avg_confidence,
            'tracking_active': self.lock_bbox is not None,
            'session_id': self.session_id
        }