"""
app.py — Versão IA Pura
----------------------
Music Cutter — Especialista em Fim de Música com IA
"""

import threading
import os
import queue
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

SUPPORTED_FORMATS = {".mp3", ".wav", ".flac", ".m4a"}

# Configurações visuais
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

COLORS = {
    "bg_dark":     "#0e0e14",
    "bg_card":     "#16161f",
    "bg_input":    "#1c1c2a",
    "accent":      "#6c63ff",
    "accent_hover":"#8a83ff",
    "success":     "#3ecf8e",
    "warning":     "#f5a623",
    "error":       "#ff5e63",
    "text_primary":"#f0f0fa",
    "text_muted":  "#7a7a9a",
    "border":      "#2a2a3d",
}

class LabeledEntry(ctk.CTkFrame):
    def __init__(self, parent, label: str, default_value: str, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.columnconfigure(0, weight=1)
        lbl = ctk.CTkLabel(self, text=label, font=ctk.CTkFont(family="Segoe UI", size=12), text_color=COLORS["text_muted"], anchor="w")
        lbl.grid(row=0, column=0, sticky="w", pady=(0, 4))
        self._var = ctk.StringVar(value=default_value)
        self._entry = ctk.CTkEntry(self, textvariable=self._var, fg_color=COLORS["bg_input"], border_color=COLORS["border"], text_color=COLORS["text_primary"], font=ctk.CTkFont(family="Segoe UI", size=13), height=38)
        self._entry.grid(row=1, column=0, sticky="ew")
    def get(self) -> str: return self._var.get()

class MusicCutterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🎙 Music Cutter IA — Edição Especial Rádio")
        icon_path = Path(__file__).with_name("assets") / "music-cutter-icon.ico"
        if icon_path.exists():
            self.iconbitmap(str(icon_path))
        self.geometry("800x600")
        self.minsize(700, 500)
        self.configure(fg_color=COLORS["bg_dark"])

        self._source_dir: Path | None = None
        self._dest_dir: Path | None = None
        self._processing = False
        self._log_queue: queue.Queue = queue.Queue()

        self._build_ui()
        self._poll_log_queue()

    def _build_ui(self):
        outer = ctk.CTkFrame(self, fg_color="transparent")
        outer.pack(fill="both", expand=True, padx=30, pady=30)
        outer.columnconfigure(0, weight=1)

        # Header
        ctk.CTkLabel(outer, text="🎙 Music Cutter IA", font=ctk.CTkFont(size=32, weight="bold"), text_color=COLORS["accent"]).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(outer, text="Processamento inteligente via YAMNet (TensorFlow)", font=ctk.CTkFont(size=14), text_color=COLORS["text_muted"]).grid(row=1, column=0, sticky="w", pady=(0, 20))

        # Card: Pastas
        self._build_folder_section(outer)

        # Card: Configuração IA
        self._build_config_section(outer)

        # Botão
        self._btn_process = ctk.CTkButton(outer, text="🚀  INICIAR CORTE INTELIGENTE", command=self._on_start, fg_color=COLORS["success"], hover_color="#2fa870", font=ctk.CTkFont(size=16, weight="bold"), height=54, corner_radius=10)
        self._btn_process.grid(row=4, column=0, sticky="ew", pady=20)

        # Log
        self._log_text = ctk.CTkTextbox(outer, font=ctk.CTkFont(family="Consolas", size=12), fg_color=COLORS["bg_card"], border_color=COLORS["border"], border_width=1)
        self._log_text.grid(row=5, column=0, sticky="nsew", pady=(0, 10))
        outer.rowconfigure(5, weight=1)

    def _build_folder_section(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=COLORS["bg_card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
        frame.grid(row=2, column=0, sticky="ew", pady=(0, 15), padx=2)
        frame.columnconfigure(1, weight=1)

        ctk.CTkButton(frame, text="Pasta Origem", command=self._pick_source, width=140, fg_color=COLORS["accent"]).grid(row=0, column=0, padx=15, pady=15)
        self._lbl_source = ctk.CTkLabel(frame, text="Selecione onde estão as músicas...", text_color=COLORS["text_muted"], anchor="w")
        self._lbl_source.grid(row=0, column=1, sticky="ew")

        ctk.CTkButton(frame, text="Pasta Destino", command=self._pick_dest, width=140, fg_color=COLORS["accent"]).grid(row=1, column=0, padx=15, pady=(0, 15))
        self._lbl_dest = ctk.CTkLabel(frame, text="Selecione para onde as músicas cortadas vão...", text_color=COLORS["text_muted"], anchor="w")
        self._lbl_dest.grid(row=1, column=1, sticky="ew")

    def _build_config_section(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=COLORS["bg_card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
        frame.grid(row=3, column=0, sticky="ew", pady=(0, 5), padx=2)
        
        self._entry_sobra = LabeledEntry(frame, label="Sobra/Mixagem (ms)", default_value="3000")
        self._entry_sobra.pack(side="left", padx=20, pady=20, fill="x", expand=True)
        
        ctk.CTkLabel(frame, text="A IA cortará exatamente no fim da música.\nA sobra será aplicada como fade ou silêncio conforme detectado.", text_color=COLORS["text_muted"], font=ctk.CTkFont(size=11), justify="left").pack(side="right", padx=20)

    def _pick_source(self):
        p = filedialog.askdirectory()
        if p: self._source_dir = Path(p); self._lbl_source.configure(text=str(p), text_color=COLORS["text_primary"])
    def _pick_dest(self):
        p = filedialog.askdirectory()
        if p: self._dest_dir = Path(p); self._lbl_dest.configure(text=str(p), text_color=COLORS["text_primary"])

    def _on_start(self):
        if not self._source_dir or not self._dest_dir:
            messagebox.showerror("Erro", "Selecione as pastas de origem e destino."); return
        
        files = [f for f in self._source_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS]
        if not files: messagebox.showwarning("Aviso", "Nenhum arquivo de áudio encontrado."); return

        self._processing = True
        self._btn_process.configure(state="disabled", text="⏳ PROCESSANDO COM IA...")
        self._clear_log()
        
        sobra = int(self._entry_sobra.get())
        threading.Thread(target=self._worker, args=(files, sobra), daemon=True).start()

    def _worker(self, files, sobra):
        self._log_queue.put("Carregando motor de IA...")
        from ai_processor import process_audio_ai

        for idx, f in enumerate(files, 1):
            self._log_queue.put(f"[{idx}/{len(files)}] {f.name}")
            dest = self._dest_dir / f.name
            ok = process_audio_ai(f, dest, sobra_ms=sobra, log_callback=lambda m: self._log_queue.put(m))
        self._log_queue.put("\n✅ TODOS OS ARQUIVOS PROCESSADOS!")
        self._log_queue.put("DONE")

    def _poll_log_queue(self):
        try:
            while True:
                m = self._log_queue.get_nowait()
                if m == "DONE":
                    self._processing = False
                    self._btn_process.configure(state="normal", text="🚀  INICIAR CORTE INTELIGENTE")
                else:
                    self._log(m)
        except queue.Empty: pass
        self.after(100, self._poll_log_queue)

    def _log(self, m):
        self._log_text.configure(state="normal")
        self._log_text.insert("end", str(m) + "\n")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _clear_log(self):
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")

if __name__ == "__main__":
    MusicCutterApp().mainloop()
