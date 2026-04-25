"""
audio_processor.py
------------------
Módulo de processamento de áudio para o Music Cutter.
Responsável por toda a lógica de corte, fade e exportação.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _patch_subprocess_no_console():
    """Evita janelas de terminal ao chamar FFmpeg/FFprobe no Windows."""
    if os.name != "nt" or getattr(subprocess.Popen, "_music_cutter_no_console", False):
        return

    original_popen = subprocess.Popen

    def quiet_popen(*args, **kwargs):
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW
        startupinfo = kwargs.get("startupinfo")
        if startupinfo is None:
            startupinfo = subprocess.STARTUPINFO()
            kwargs["startupinfo"] = startupinfo
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        return original_popen(*args, **kwargs)

    quiet_popen._music_cutter_no_console = True
    subprocess.Popen = quiet_popen


_patch_subprocess_no_console()

# ── Detecta e configura o FFmpeg no PATH antes do pydub ──────────────────────────
def _setup_ffmpeg_path():
    """Busca o FFmpeg e adiciona ao PATH do processo para o pydub encontrar."""
    candidates = [
        # Caminho onde instalamos via winget
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages",
        Path(r"C:\ffmpeg\bin"),
        Path(r"C:\Program Files\ffmpeg\bin"),
    ]

    bin_dir = None
    # 1. Procura no WinGet (com rglob para ser preciso)
    winget_base = candidates[0]
    if winget_base.exists():
        ffmpeg_exe = next(winget_base.rglob("ffmpeg.exe"), None)
        if ffmpeg_exe:
            bin_dir = ffmpeg_exe.parent

    # 2. Se não achou, checa caminhos fixos
    if not bin_dir:
        for p in candidates[1:]:
            if (p / "ffmpeg.exe").exists():
                bin_dir = p
                break

    if bin_dir:
        bin_path = str(bin_dir)
        # Adiciona ao PATH do Windows para este processo
        if bin_path not in os.environ["PATH"]:
            os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
        return bin_path
    
    return None

_FFMPEG_BIN = _setup_ffmpeg_path()

# Configuração explícita do pydub (previne warnings se importado depois)
try:
    import pydub
    if _FFMPEG_BIN:
        pydub.AudioSegment.converter = os.path.join(_FFMPEG_BIN, "ffmpeg.exe")
        pydub.AudioSegment.ffprobe = os.path.join(_FFMPEG_BIN, "ffprobe.exe")
except ImportError:
    pass

from pydub import AudioSegment
from pydub.utils import mediainfo
# ─────────────────────────────────────────────────────────────────────────────



SUPPORTED_FORMATS = {".mp3", ".wav", ".flac"}


def get_ffmpeg_status() -> str:
    """Retorna uma string descrevendo o status do FFmpeg para exibir no log."""
    if _FFMPEG_BIN:
        return f"FFmpeg encontrado em: {_FFMPEG_BIN}"
    return "⚠ FFmpeg NÃO encontrado no PATH. Instale via: winget install Gyan.FFmpeg"


def find_start_ms(audio: AudioSegment, threshold_dbfs: float, chunk_ms: int = 10) -> int:
    """
    Varre o áudio do início para o fim em janelas de `chunk_ms` ms
    e retorna o timestamp (ms) onde o volume ultrapassa `threshold_dbfs`.
    Retorna 0 se o threshold nunca for atingido.
    """
    for i in range(0, len(audio), chunk_ms):
        chunk = audio[i: i + chunk_ms]
        if len(chunk) == 0:
            break
        if chunk.dBFS > threshold_dbfs:
            return max(0, i - chunk_ms)  # um chunk antes para não cortar o ataque
    return 0


def find_end_ms(audio: AudioSegment, threshold_dbfs: float, chunk_ms: int = 10) -> int:
    """
    Varre o áudio do fim para o início em janelas de `chunk_ms` ms
    e retorna o timestamp (ms) onde o volume sobe acima de `threshold_dbfs`.
    Retorna len(audio) se o threshold nunca for encontrado.
    """
    total_ms = len(audio)
    for i in range(total_ms, 0, -chunk_ms):
        start = max(0, i - chunk_ms)
        chunk = audio[start:i]
        if len(chunk) == 0:
            continue
        if chunk.dBFS > threshold_dbfs:
            return min(total_ms, i + chunk_ms)  # um chunk depois para não cortar o decaimento
    return total_ms


def get_export_params(source_path: Path) -> dict:
    """
    Extrai parâmetros de exportação do arquivo original para manter a qualidade.
    Retorna um dicionário compatível com AudioSegment.export().
    """
    ext = source_path.suffix.lower().lstrip(".")
    if ext == "mp3":
        try:
            info = mediainfo(str(source_path))
            bitrate = info.get("bit_rate", "192000")
            # Converte para kbps inteiro
            bitrate_kbps = int(int(bitrate) / 1000)
            bitrate_str = f"{bitrate_kbps}k"
        except Exception:
            bitrate_str = "192k"
        return {"format": "mp3", "bitrate": bitrate_str}

    elif ext == "flac":
        return {"format": "flac"}

    elif ext == "wav":
        return {"format": "wav"}

    return {"format": ext}


def _get_ffmpeg_executable() -> str:
    """Retorna o executavel do FFmpeg configurado para o pydub."""
    if _FFMPEG_BIN:
        return os.path.join(_FFMPEG_BIN, "ffmpeg.exe")
    return getattr(AudioSegment, "converter", None) or "ffmpeg"


def export_with_original_metadata(
    audio: AudioSegment,
    source_path: Path,
    dest_path: Path,
    export_params: dict,
    log_callback=None,
) -> None:
    """
    Exporta o audio processado e remuxa metadados/capa do arquivo original.

    O pydub reencoda o audio, mas nao preserva arte embutida nem todos os
    campos de metadata. Por isso geramos um temporario e deixamos o FFmpeg
    copiar metadata global, capitulos e streams de imagem anexada do original.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_file = tempfile.NamedTemporaryFile(
        prefix=f".{dest_path.stem}.audio-",
        suffix=dest_path.suffix,
        dir=dest_path.parent,
        delete=False,
    )
    tmp_path = Path(tmp_file.name)
    tmp_file.close()

    def log(msg: str):
        if log_callback:
            log_callback(msg)

    try:
        audio.export(str(tmp_path), **export_params)

        ffmpeg = _get_ffmpeg_executable()
        cmd = [
            ffmpeg,
            "-y",
            "-i", str(tmp_path),
            "-i", str(source_path),
            "-map", "0:a:0",
            "-map", "1:v?",
            "-map_metadata", "1",
            "-map_chapters", "1",
            "-c:a", "copy",
            "-c:v", "copy",
        ]

        if dest_path.suffix.lower() == ".mp3":
            cmd.extend(["-id3v2_version", "3", "-write_id3v1", "1"])

        cmd.append(str(dest_path))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        if result.returncode != 0:
            log("  ! Aviso: nao foi possivel copiar capa/metadados via FFmpeg; exportando audio sem remux.\n")
            audio.export(str(dest_path), **export_params)

    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except TypeError:
            if tmp_path.exists():
                tmp_path.unlink()


def process_audio_file(
    source_path: Path,
    dest_path: Path,
    start_threshold_dbfs: float,
    end_threshold_dbfs: float,
    fade_in_ms: int,
    fade_out_ms: int,
    silence_end_ms: int,
    log_callback=None,
) -> bool:
    """
    Processa um único arquivo de áudio:
      1. Carrega o arquivo.
      2. Remove silêncio/ruído do início.
      3. Remove silêncio/ruído do fim.
      4. Aplica fade-in e fade-out.
      5. Adiciona silêncio absoluto no final.
      6. Exporta mantendo formato e qualidade originais.

    Retorna True em sucesso, False em falha.
    """
    def log(msg: str):
        if log_callback:
            log_callback(msg)

    try:
        ext = source_path.suffix.lower()
        fmt = ext.lstrip(".")

        log(f"  ↳ Carregando: {source_path.name}")
        audio = AudioSegment.from_file(str(source_path), format=fmt)

        original_duration = len(audio) / 1000
        log(f"  ↳ Duração original: {original_duration:.2f}s")

        # --- 1. Cortar início ---
        start_ms = find_start_ms(audio, start_threshold_dbfs)
        # --- 2. Cortar fim ---
        end_ms = find_end_ms(audio, end_threshold_dbfs)

        if start_ms >= end_ms:
            log(f"  ⚠ Aviso: O resultado do corte está vazio (start={start_ms}ms >= end={end_ms}ms). "
                f"Verifique os limiares. Arquivo ignorado.")
            return False

        audio = audio[start_ms:end_ms]
        trimmed_duration = len(audio) / 1000
        log(f"  ↳ Cortado: início em {start_ms}ms | fim em {end_ms}ms | nova duração: {trimmed_duration:.2f}s")

        # --- 3. Fade-in / Fade-out ---
        if fade_in_ms > 0:
            audio = audio.fade_in(min(fade_in_ms, len(audio)))
        if fade_out_ms > 0:
            audio = audio.fade_out(min(fade_out_ms, len(audio)))

        # --- 4. Silêncio no final ---
        if silence_end_ms > 0:
            silence = AudioSegment.silent(duration=silence_end_ms)
            audio = audio + silence

        # --- 5. Exportar ---
        export_params = get_export_params(source_path)
        export_with_original_metadata(audio, source_path, dest_path, export_params, log_callback=log)

        final_duration = len(audio) / 1000
        log(f"  ✔ Exportado: {dest_path.name} | duração final: {final_duration:.2f}s\n")
        return True

    except Exception as e:
        log(f"  ✖ Erro ao processar '{source_path.name}': {e}\n")
        return False
