"""
CADIX - Ventana Principal
Interfaz de usuario principal mejorada y profesional.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time
import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
import collections

from src.core.detector import CADIXDetector
from src.core.database import DatabaseManager
from src.core.logger import get_logger
from src.config.settings import ConfigManager, SystemConfig

class CADIXMainWindow:
    """Ventana principal de CADIX"""
    
    def __init__(self):
        # Configurar tema
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        
        # Inicializar componentes
        self.logger = get_logger()
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config()
        self.db = DatabaseManager(os.path.join(self.config.base_dir, self.config.database.db_path))
        self.root = ctk.CTk()
        self.root.title("CADIX - Sistema Industrial de Detección de Cadenas v2.0")
        self.root.geometry(f"{self.config.ui.window_width}x{self.config.ui.window_height}")
        self.root.minsize(1200, 700)
        
        # Variables de estado
        self.detector: Optional[CADIXDetector] = None
        self.current_session: Optional[str] = None
        self.operator_name = tk.StringVar(master=self.root, value="Operador")
        self.shift_name = tk.StringVar(master=self.root, value="Turno Mañana")
        
        # Métricas en tiempo real
        self.last_measurement = tk.StringVar(master=self.root, value="—")
        self.ema_measurement = tk.StringVar(master=self.root, value="—")
        self.avg_measurement = tk.StringVar(master=self.root, value="—")
        self.current_fps = tk.StringVar(master=self.root, value="—")
        self.detection_count = tk.StringVar(master=self.root, value="0")
        self.alert_count = tk.StringVar(master=self.root, value="0")
        self.session_time = tk.StringVar(master=self.root, value="00:00:00")
        
        # Buffer de mediciones recientes
        self.recent_shots = collections.deque(maxlen=10)
        self.measurement_history = []
        
        # Ventana principal
        
        # Variables de UI
        self.video_frame = None
        self.status_text = tk.StringVar(master=self.root, value="Sistema listo")
        
        self._build_interface()
        self._setup_session_timer()
        self.logger.log_system_start(self.config.version)
    
    def _build_interface(self):
        """Construye la interfaz principal"""
        # Panel superior - Barra de herramientas
        self._build_toolbar()
        
        # Panel principal - Dividido en 3 columnas
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Columna izquierda - Controles
        self._build_left_panel()
        
        # Columna central - Video
        self._build_video_panel()
        
        # Columna derecha - Métricas y datos
        self._build_right_panel()
        
        # Panel inferior - Barra de estado
        self._build_status_bar()
    
    def _build_toolbar(self):
        """Construye la barra de herramientas superior"""
        toolbar = ctk.CTkFrame(self.root, height=60)
        toolbar.pack(fill="x", padx=10, pady=(10, 5))
        
        # Logo y título
        title_frame = ctk.CTkFrame(toolbar)
        title_frame.pack(side="left", fill="y", padx=(10, 20))
        
        ctk.CTkLabel(
            title_frame, 
            text="CADIX", 
            font=ctk.CTkFont(size=24, weight="bold")
        ).pack(side="left", padx=10, pady=10)
        
        ctk.CTkLabel(
            title_frame,
            text="Sistema Industrial v2.0",
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=(0, 10), pady=10)
        
        # Controles de sesión
        session_frame = ctk.CTkFrame(toolbar)
        session_frame.pack(side="left", fill="y", padx=10)
        
        ctk.CTkLabel(session_frame, text="Operador:").grid(row=0, column=0, padx=5, pady=2, sticky="e")
        ctk.CTkEntry(
            session_frame, 
            textvariable=self.operator_name, 
            width=120
        ).grid(row=0, column=1, padx=5, pady=2)
        
        ctk.CTkLabel(session_frame, text="Turno:").grid(row=1, column=0, padx=5, pady=2, sticky="e")
        shift_combo = ctk.CTkComboBox(
            session_frame,
            variable=self.shift_name,
            values=["Turno Mañana", "Turno Tarde", "Turno Noche"],
            width=120
        )
        shift_combo.grid(row=1, column=1, padx=5, pady=2)
        
        # Botones principales
        button_frame = ctk.CTkFrame(toolbar)
        button_frame.pack(side="right", fill="y", padx=10)
        
        self.start_btn = ctk.CTkButton(
            button_frame,
            text="▶ INICIAR",
            command=self._start_detection,
            fg_color="green",
            width=100
        )
        self.start_btn.pack(side="left", padx=5, pady=10)
        
        self.stop_btn = ctk.CTkButton(
            button_frame,
            text="⏹ DETENER",
            command=self._stop_detection,
            fg_color="red",
            width=100,
            state="disabled"
        )
        self.stop_btn.pack(side="left", padx=5, pady=10)
    
    def _build_left_panel(self):
        """Panel izquierdo con controles"""
        left_panel = ctk.CTkFrame(self.main_frame, width=280)
        left_panel.pack(side="left", fill="y", padx=(0, 5))
        left_panel.pack_propagate(False)
        
        # Información de sesión
        session_info = ctk.CTkFrame(left_panel)
        session_info.pack(fill="x", padx=10, pady=(10, 5))
        
        ctk.CTkLabel(
            session_info, 
            text="INFORMACIÓN DE SESIÓN", 
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        # Tiempo de sesión
        time_frame = ctk.CTkFrame(session_info)
        time_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(time_frame, text="Tiempo activo:").pack()
        ctk.CTkLabel(
            time_frame, 
            textvariable=self.session_time,
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack()
        
        # Configuración rápida
        config_frame = ctk.CTkFrame(left_panel)
        config_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(
            config_frame, 
            text="CONFIGURACIÓN RÁPIDA", 
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        # Umbral de alerta
        threshold_frame = ctk.CTkFrame(config_frame)
        threshold_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(threshold_frame, text="Umbral de alerta (mm):").pack()
        self.threshold_entry = ctk.CTkEntry(
            threshold_frame,
            width=100
        )
        self.threshold_entry.pack(pady=2)
        self.threshold_entry.insert(0, str(self.config.measurement.alert_threshold_mm))
        
        # Factor de conversión
        conversion_frame = ctk.CTkFrame(config_frame)
        conversion_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(conversion_frame, text="Conversión px→mm:").pack()
        self.conversion_entry = ctk.CTkEntry(
            conversion_frame,
            width=100
        )
        self.conversion_entry.pack(pady=2)
        self.conversion_entry.insert(0, f"{self.config.measurement.px_to_mm:.3f}")
        
        ctk.CTkButton(
            config_frame,
            text="Aplicar cambios",
            command=self._apply_quick_config
        ).pack(pady=10)
        
        # Botones de acción
        actions_frame = ctk.CTkFrame(left_panel)
        actions_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(
            actions_frame, 
            text="ACCIONES", 
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        buttons = [
            ("⚙ Configuración", self._open_settings),
            ("📊 Reportes", self._generate_report),
            ("📁 Exportar datos", self._export_data),
            ("🔧 Calibración", self._open_calibration),
            ("💾 Backup", self._create_backup)
        ]
        
        for text, command in buttons:
            ctk.CTkButton(
                actions_frame,
                text=text,
                command=command,
                width=200
            ).pack(pady=3, padx=10)
    
    def _build_video_panel(self):
        """Panel central con video"""
        video_panel = ctk.CTkFrame(self.main_frame)
        video_panel.pack(side="left", fill="both", expand=True, padx=5)
        
        # Título
        ctk.CTkLabel(
            video_panel,
            text="VISTA DE CÁMARA",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        # Frame para el video
        self.video_frame = ctk.CTkFrame(video_panel, fg_color="black")
        self.video_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))
        
        # Label para mostrar video
        self.video_label = ctk.CTkLabel(
            self.video_frame,
            text="Cámara desconectada\nPresiona INICIAR para comenzar",
            font=ctk.CTkFont(size=18)
        )
        self.video_label.pack(expand=True, fill="both")
    
    def _build_right_panel(self):
        """Panel derecho con métricas"""
        right_panel = ctk.CTkFrame(self.main_frame, width=320)
        right_panel.pack(side="right", fill="y", padx=(5, 0))
        right_panel.pack_propagate(False)
        
        # Métricas en tiempo real
        self._build_metrics_panel(right_panel)
        
        # Gráfico de historial
        self._build_chart_panel(right_panel)
        
        # Últimas mediciones
        self._build_recent_measurements_panel(right_panel)
    
    def _build_metrics_panel(self, parent):
        """Panel de métricas en tiempo real"""
        metrics_frame = ctk.CTkFrame(parent)
        metrics_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        ctk.CTkLabel(
            metrics_frame,
            text="MÉTRICAS EN TIEMPO REAL",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        # Grid de métricas
        grid_frame = ctk.CTkFrame(metrics_frame)
        grid_frame.pack(fill="x", padx=10, pady=(5, 10))
        
        metrics = [
            ("Última medición:", self.last_measurement, "mm"),
            ("Promedio EMA:", self.ema_measurement, "mm"),
            ("Promedio sesión:", self.avg_measurement, "mm"),
            ("FPS actual:", self.current_fps, "fps"),
            ("Detecciones:", self.detection_count, ""),
            ("Alertas:", self.alert_count, "")
        ]
        
        for i, (label, var, unit) in enumerate(metrics):
            row = i // 2
            col = (i % 2) * 2
            
            ctk.CTkLabel(grid_frame, text=label).grid(
                row=row, column=col, sticky="w", padx=5, pady=2
            )
            
            value_text = f"{var.get()}{' ' + unit if unit else ''}"
            ctk.CTkLabel(
                grid_frame,
                text=value_text,
                font=ctk.CTkFont(weight="bold")
            ).grid(row=row, column=col+1, sticky="e", padx=5, pady=2)
    
    def _build_chart_panel(self, parent):
        """Panel con gráfico de historial"""
        chart_frame = ctk.CTkFrame(parent)
        chart_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(
            chart_frame,
            text="HISTORIAL DE MEDICIONES",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        # Crear figura matplotlib
        self.fig = Figure(figsize=(5, 3), dpi=80)
        self.fig.patch.set_facecolor('#212121')
        
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#2b2b2b')
        self.ax.tick_params(colors='white')
        self.ax.set_xlabel('Tiempo', color='white')
        self.ax.set_ylabel('mm', color='white')
        self.ax.grid(True, alpha=0.3, color='gray')
        
        # Línea de datos
        self.line, = self.ax.plot([], [], 'cyan', linewidth=1.5)
        self.threshold_line = self.ax.axhline(
            y=self.config.measurement.alert_threshold_mm,
            color='red', linestyle='--', alpha=0.7
        )
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(5, 10))
    
    def _build_recent_measurements_panel(self, parent):
        """Panel de mediciones recientes"""
        recent_frame = ctk.CTkFrame(parent)
        recent_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        ctk.CTkLabel(
            recent_frame,
            text="ÚLTIMAS MEDICIONES",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=(10, 5))
        
        # Lista de mediciones
        self.measurements_listbox = tk.Listbox(
            recent_frame,
            bg='#2b2b2b',
            fg='white',
            selectbackground='#1f538d',
            height=8,
            font=('Consolas', 9)
        )
        self.measurements_listbox.pack(fill="both", expand=True, padx=10, pady=(5, 10))
        
        # Cargar mediciones recientes desde la base de datos
        self._load_recent_measurements()
    
    def _build_status_bar(self):
        """Barra de estado inferior"""
        status_frame = ctk.CTkFrame(self.root, height=30)
        status_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self.status_label = ctk.CTkLabel(
            status_frame,
            textvariable=self.status_text,
            anchor="w"
        )
        self.status_label.pack(side="left", padx=10, pady=5)
        
        # Indicadores de estado
        self.indicators_frame = ctk.CTkFrame(status_frame)
        self.indicators_frame.pack(side="right", padx=10, pady=2)
        
        self.camera_indicator = ctk.CTkLabel(
            self.indicators_frame,
            text="📹 Desconectado",
            width=100
        )
        self.camera_indicator.pack(side="left", padx=5)
        
        self.db_indicator = ctk.CTkLabel(
            self.indicators_frame,
            text="💾 Conectado",
            width=100
        )
        self.db_indicator.pack(side="left", padx=5)
    
    def _setup_session_timer(self):
        """Configura el temporizador de sesión"""
        self.session_start_time = None
        self._update_session_timer()
    
    def _update_session_timer(self):
        """Actualiza el temporizador de sesión"""
        if self.session_start_time and self.current_session:
            elapsed = time.time() - self.session_start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            self.session_time.set(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        else:
            self.session_time.set("00:00:00")
        
        self.root.after(1000, self._update_session_timer)
    
    def _start_detection(self):
        """Inicia el sistema de detección"""
        try:
            if self.detector and self.detector.is_alive():
                messagebox.showwarning("Sistema activo", "El detector ya está en funcionamiento")
                return
            
            # Validar configuración
            if not os.path.exists(self.config.detection.model_path):
                messagebox.showerror(
                    "Error", 
                    f"Modelo no encontrado:\n{self.config.detection.model_path}"
                )
                return
            
            # Iniciar sesión
            self.current_session = self.db.start_session(
                operator=self.operator_name.get(),
                shift=self.shift_name.get()
            )
            self.session_start_time = time.time()
            
            # Crear detector
            self.detector = CADIXDetector(
                config=self.config,
                frame_callback=self._on_frame_update,
                stats_callback=self._on_stats_update,
                status_callback=self._on_status_update,
                shot_callback=self._on_shot_measurement,
                session_id=self.current_session
            )
            
            # Iniciar detector
            self.detector.start()
            
            # Actualizar UI
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.camera_indicator.configure(text="📹 Conectado", text_color="green")
            
            self.logger.info(f"Sistema iniciado - Sesión: {self.current_session}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error iniciando detección:\n{str(e)}")
            self.logger.error(f"Error iniciando detección: {e}")
    
    def _stop_detection(self):
        """Detiene el sistema de detección"""
        try:
            if self.detector:
                self.detector.stop()
                self.detector = None
            
            # Finalizar sesión
            if self.current_session:
                self.db.end_session(self.current_session)
                self.current_session = None
                self.session_start_time = None
            
            # Actualizar UI
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.camera_indicator.configure(text="📹 Desconectado", text_color="gray")
            
            # Limpiar video
            self.video_label.configure(
                image=None,
                text="Cámara desconectada\nPresiona INICIAR para comenzar"
            )
            
            self.logger.info("Sistema detenido")
            
        except Exception as e:
            self.logger.error(f"Error deteniendo detección: {e}")
    
    def _on_frame_update(self, bgr_frame):
        """Callback para actualización de frames"""
        try:
            # Redimensionar frame para UI
            height, width = bgr_frame.shape[:2]
            aspect_ratio = width / height
            
            display_width = 640
            display_height = int(display_width / aspect_ratio)
            
            import cv2 as cv
            frame_resized = cv.resize(bgr_frame, (display_width, display_height))
            frame_rgb = cv.cvtColor(frame_resized, cv.COLOR_BGR2RGB)
            
            # Convertir a PhotoImage
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image)
            
            # Actualizar label
            self.video_label.configure(image=photo, text="")
            self.video_label.image = photo  # Mantener referencia
            
        except Exception as e:
            self.logger.error(f"Error actualizando frame: {e}")
    
    def _on_stats_update(self, last_mm, ema_mm, avg_mm, update_plot=False):
        """Callback para actualización de estadísticas"""
        try:
            self.last_measurement.set(f"{last_mm:.1f}")
            self.ema_measurement.set(f"{ema_mm:.1f}")
            self.avg_measurement.set(f"{avg_mm:.1f}")
            
            # Actualizar historial
            self.measurement_history.append(last_mm)
            if len(self.measurement_history) > 200:
                self.measurement_history = self.measurement_history[-200:]
            
            # Actualizar gráfico
            if update_plot and self.measurement_history:
                self.line.set_data(range(len(self.measurement_history)), self.measurement_history)
                self.ax.relim()
                self.ax.autoscale_view()
                self.canvas.draw_idle()
            
            # Guardar en base de datos (cada 10 mediciones para no saturar)
            if len(self.measurement_history) % 10 == 0:
                self.db.save_measurement(
                    value_mm=last_mm,
                    ema_value=ema_mm,
                    session_id=self.current_session
                )
        
        except Exception as e:
            self.logger.error(f"Error actualizando estadísticas: {e}")
    
    def _on_status_update(self, status_text):
        """Callback para actualización de estado"""
        self.status_text.set(status_text)
    
    def _on_shot_measurement(self, value_mm, direction, session_id):
        """Callback para mediciones one-shot"""
        try:
            # Guardar en base de datos
            self.db.save_oneshot(
                value_mm=value_mm,
                direction=direction,
                session_id=session_id,
                operator=self.operator_name.get(),
                shift=self.shift_name.get()
            )
            
            # Actualizar UI
            timestamp = datetime.now().strftime("%H:%M:%S")
            measurement_text = f"[{timestamp}] {value_mm:.1f}mm ({direction})"
            
            self.measurements_listbox.insert(0, measurement_text)
            
            # Mantener solo últimas 20 mediciones en la lista
            if self.measurements_listbox.size() > 20:
                self.measurements_listbox.delete(self.measurements_listbox.size()-1)
            
            self.recent_shots.appendleft({
                'timestamp': timestamp,
                'value': value_mm,
                'direction': direction
            })
            
            self.logger.info(f"Medición one-shot registrada: {value_mm:.1f}mm ({direction})")
            
        except Exception as e:
            self.logger.error(f"Error procesando medición one-shot: {e}")
    
    def _load_recent_measurements(self):
        """Carga mediciones recientes desde la base de datos"""
        try:
            recent = self.db.get_recent_oneshots(limit=20)
            
            self.measurements_listbox.delete(0, tk.END)
            
            for measurement in recent:
                timestamp = datetime.fromisoformat(measurement['timestamp']).strftime("%H:%M:%S")
                text = f"[{timestamp}] {measurement['value_mm']:.1f}mm ({measurement['direction']})"
                self.measurements_listbox.insert(tk.END, text)
        
        except Exception as e:
            self.logger.error(f"Error cargando mediciones recientes: {e}")
    
    def _apply_quick_config(self):
        """Aplica configuración rápida"""
        try:
            # Actualizar configuración
            new_threshold = float(self.threshold_entry.get())
            new_conversion = float(self.conversion_entry.get())
            
            self.config.measurement.alert_threshold_mm = new_threshold
            self.config.measurement.px_to_mm = new_conversion
            
            # Guardar configuración
            self.config_manager.save_config(self.config)
            
            # Actualizar gráfico si existe
            if hasattr(self, 'threshold_line'):
                self.threshold_line.set_ydata([new_threshold, new_threshold])
                self.canvas.draw_idle()
            
            messagebox.showinfo("Configuración", "Configuración aplicada correctamente")
            self.logger.info(f"Configuración actualizada: umbral={new_threshold}mm, conversión={new_conversion}")
            
        except ValueError:
            messagebox.showerror("Error", "Valores de configuración inválidos")
        except Exception as e:
            messagebox.showerror("Error", f"Error aplicando configuración:\n{str(e)}")
            self.logger.error(f"Error aplicando configuración: {e}")
    
    def _open_settings(self):
        """Abre ventana de configuración avanzada"""
        # TODO: Implementar ventana de configuración completa
        messagebox.showinfo("Configuración", "Ventana de configuración avanzada en desarrollo")
    
    def _generate_report(self):
        """Genera reporte de la sesión actual"""
        # TODO: Implementar generación de reportes
        messagebox.showinfo("Reportes", "Generador de reportes en desarrollo")
    
    def _export_data(self):
        """Exporta datos a CSV"""
        try:
            if not self.current_session:
                messagebox.showwarning("Sin sesión", "No hay sesión activa para exportar")
                return
            
            # Seleccionar archivo
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfilename=f"cadix_export_{self.current_session}.csv"
            )
            
            if filename:
                # TODO: Implementar exportación real
                messagebox.showinfo("Exportación", f"Datos exportados a:\n{filename}")
                self.logger.info(f"Datos exportados a: {filename}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error exportando datos:\n{str(e)}")
            self.logger.error(f"Error exportando datos: {e}")
    
    def _open_calibration(self):
        """Abre herramienta de calibración"""
        # TODO: Implementar calibración avanzada
        messagebox.showinfo("Calibración", "Herramienta de calibración en desarrollo")
    
    def _create_backup(self):
        """Crea backup de la base de datos"""
        try:
            success = self.db.backup_database()
            if success:
                messagebox.showinfo("Backup", "Backup creado exitosamente")
                self.logger.info("Backup de base de datos creado")
            else:
                messagebox.showerror("Error", "Error creando backup")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error creando backup:\n{str(e)}")
            self.logger.error(f"Error creando backup: {e}")
    
    def _update_detector_metrics(self):
        """Actualiza métricas del detector"""
        if self.detector:
            try:
                stats = self.detector.get_stats()
                self.current_fps.set(f"{stats['fps']:.1f}")
                self.detection_count.set(str(stats['detection_count']))
                self.alert_count.set(str(stats['alert_count']))
            except Exception as e:
                self.logger.error(f"Error actualizando métricas: {e}")
        
        self.root.after(2000, self._update_detector_metrics)  # Actualizar cada 2 segundos
    
    def run(self):
        """Ejecuta la aplicación"""
        try:
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
            self._update_detector_metrics()  # Iniciar actualización de métricas
            self.root.mainloop()
        except Exception as e:
            self.logger.error(f"Error en aplicación principal: {e}")
    
    def _on_closing(self):
        """Maneja el cierre de la aplicación"""
        try:
            if self.detector:
                self._stop_detection()
            
            self.logger.log_system_shutdown()
            self.root.destroy()
        
        except Exception as e:
            self.logger.error(f"Error cerrando aplicación: {e}")
            self.root.destroy()