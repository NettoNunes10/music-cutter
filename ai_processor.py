import os
import re
import shutil

# Silenciar logs do TensorFlow antes de importar o runtime.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from pathlib import Path
from audio_processor import export_with_original_metadata
from pydub import AudioSegment
from pydub.utils import mediainfo

tf = None
hub = None

# Configurações globais
YAMNET_MODEL_URL = 'https://tfhub.dev/google/yamnet/1'
YAMNET_MODEL = None
TFHUB_BAD_MODEL_MSG = "contains neither 'saved_model.pb' nor 'saved_model.pbtxt'"

# Índices Oficiais do YAMNet (Confirmados via mapeamento AudioSet)
ID_SPEECH = 0
ID_MUSIC = 132
ID_SINGING = 24
ID_A_CAPPELLA = 250
ID_APPLAUSE = 62
ID_CHEERING = 61
ID_CROWD = 64

def _find_bad_tfhub_cache_path(error: Exception) -> Path | None:
    match = re.search(r"'([^']*tfhub_modules[^']*)'", str(error))
    if not match:
        return None
    cache_path = Path(match.group(1))
    return cache_path if cache_path.exists() else None


def _remove_bad_tfhub_cache(error: Exception, log_callback=None) -> bool:
    cache_path = _find_bad_tfhub_cache_path(error)
    if not cache_path:
        return False

    def log(m):
        if log_callback:
            log_callback(m)

    try:
        shutil.rmtree(cache_path)
        log(f"  ↳ Cache corrompido do YAMNet removido: {cache_path}")
        return True
    except Exception as cleanup_error:
        log(f"  ⚠ Não consegui remover o cache corrompido do YAMNet: {cleanup_error}")
        return False


def load_yamnet(log_callback=None):
    """Carrega o modelo YAMNet de forma global."""
    global YAMNET_MODEL, tf, hub
    if YAMNET_MODEL is None:
        if hub is None:
            import tensorflow as _tf
            import tensorflow_hub as _hub
            tf = _tf
            hub = _hub
        try:
            YAMNET_MODEL = hub.load(YAMNET_MODEL_URL)
        except ValueError as e:
            if TFHUB_BAD_MODEL_MSG not in str(e) or not _remove_bad_tfhub_cache(e, log_callback):
                raise
            if log_callback:
                log_callback("  ↳ Baixando o modelo YAMNet novamente...")
            YAMNET_MODEL = hub.load(YAMNET_MODEL_URL)
    return YAMNET_MODEL

def get_yamnet_predictions(audio_segment: AudioSegment, log_callback=None):
    """Resample para 16kHz mono e extrai probabilidades do YAMNet."""
    # Requisito YAMNet: 16kHz Mono
    audio = audio_segment.set_frame_rate(16000).set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    
    # Normalização para [-1, 1] (PCM 16-bit)
    if audio.sample_width == 2:
        samples /= 32768.0
    elif audio.sample_width == 4:
        samples /= 2147483648.0
        
    model = load_yamnet(log_callback=log_callback)
    scores, _, _ = model(samples)
    return scores.numpy()

def identify_cue_out_ms(audio: AudioSegment, log_callback=None) -> tuple[int, str]:
    """
    Engine de Decisão Baseado em Regras de Engenharia de Áudio.
    Implementa Rolling Windows e Thresholds Rígidos.
    """
    def log(m):
        if log_callback: log_callback(m)

    duration_ms = len(audio)
    # Analisamos apenas os últimos 45 segundos
    analysis_duration_ms = min(duration_ms, 45000)
    start_offset = duration_ms - analysis_duration_ms
    tail = audio[start_offset:]
    
    scores = get_yamnet_predictions(tail, log_callback=log)
    num_frames = scores.shape[0]
    frame_step_ms = 480 # Passo de 480ms entre frames (YAMNet hop)
    
    # Agrupamento de Classes Positivas (Música e derivados vocais)
    # Somamos Music + Singing + A Cappella para evitar falsos cortes em pontes vocais
    music_score = scores[:, ID_MUSIC] + scores[:, ID_SINGING] + scores[:, ID_A_CAPPELLA]
    speech_score = scores[:, ID_SPEECH]
    crowd_score = scores[:, ID_APPLAUSE] + scores[:, ID_CHEERING] + scores[:, ID_CROWD]
    
    log(f"--- Iniciando análise de {num_frames} frames ---")

    for i in range(num_frames):
        current_ms = start_offset + (i * frame_step_ms)
        
        # --- REGRA C: Fim Seco / Silêncio (Físico) ---
        chunk = tail[i * frame_step_ms : (i + 1) * frame_step_ms]
        if len(chunk) > 0 and chunk.dBFS < -45:
            log(f"  [!] Regra C: Silêncio físico em {current_ms/1000:.2f}s (RMS: {chunk.dBFS:.1f} dBFS)")
            return current_ms, "Regra_C"

        # --- REGRA A: Transição para Voz/Plateia (Rolling Window 3 frames) ---
        if i <= num_frames - 3:
            # Verifica se a condição se mantém em i, i+1 e i+2
            window_indices = range(i, i + 3)
            is_rule_a = all(
                music_score[j] < 0.20 and (speech_score[j] > 0.60 or crowd_score[j] > 0.50)
                for j in range(i, i + 3)
            )
            if is_rule_a:
                log(f"  [!] Regra A: Transição validada em {current_ms/1000:.2f}s (Buffer: 3 frames)")
                return current_ms, "Regra_A"

        # --- REGRA B: Fade-out Longo / Baixa Energia Musical (Rolling Window 4 frames) ---
        if i <= num_frames - 4:
            is_rule_b = all(music_score[j] < 0.15 for j in range(i, i + 4))
            if is_rule_b:
                log(f"  [!] Regra B: Fade-out detectado em {current_ms/1000:.2f}s (Buffer: 4 frames)")
                return current_ms, "Regra_B"

    # Fallback: Fim original do arquivo
    return duration_ms, "Fim_Original"

def process_audio_ai(source_path: Path, dest_path: Path, sobra_ms: int = 3000, log_callback=None):
    """
    Orquestrador de Processamento.
    Preserva Formato, Sample Rate e Bitrate originais.
    """
    def log(m):
        if log_callback: log_callback(m)
        else: print(m)
        
    try:
        log(f"Processing: {source_path.name}")
        audio = AudioSegment.from_file(str(source_path))
        
        # Identificação de Cue Out
        cue_out_ms, rule = identify_cue_out_ms(audio, log_callback=log)
        
        # 2. Aplicação da lógica de 'Sobra' para Mixagem (Nova Especificação)
        # O objetivo é que o arquivo final tenha a duração de [cue_out_ms + sobra_ms]
        # O trecho 'sobra_ms' deve ser processado com fade out.
        
        audio_main = audio[:cue_out_ms]
        audio_tail = audio[cue_out_ms : cue_out_ms + sobra_ms]
        
        # Se a faixa original for mais curta que a sobra necessária, completa com silêncio
        if len(audio_tail) < sobra_ms:
            padding_ms = sobra_ms - len(audio_tail)
            audio_tail += AudioSegment.silent(duration=padding_ms, frame_rate=audio.frame_rate)
            
        # Aplica o fade out exato na margem adicional
        audio_tail_faded = audio_tail.fade_out(duration=sobra_ms)
        
        # Resultado final: Música até o ponto detectado + Margem processada
        audio_final = audio_main + audio_tail_faded
        log(f"  ↳ Mixagem: Cue Out em {cue_out_ms/1000:.2f}s + {sobra_ms}ms de margem com Fade Out.")
            
        # Extração de parâmetros originais para exportação
        info = mediainfo(str(source_path))
        ext = source_path.suffix.lower().lstrip(".")
        
        # Parâmetros de preservação (Sample Rate e Tags)
        export_params = {
            "format": ext,
            "tags": info.get("TAG", {}),
            "parameters": ["-ar", info.get("sample_rate", "44100")]
        }
        
        # Tratamento de Bitrate para MP3
        if ext == "mp3":
            bitrate = info.get("bit_rate", "192000")
            # Converte para string kbps (ex: "320k")
            export_params["bitrate"] = f"{int(int(bitrate)/1000)}k"
            
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        export_with_original_metadata(audio_final, source_path, dest_path, export_params, log_callback=log)
        
        log(f"✅ Exportado com sucesso: {dest_path.name}\n")
        return True
        
    except Exception as e:
        log(f"❌ Erro Crítico: {str(e)}")
        import traceback
        log(traceback.format_exc())
        return False

# Formatos para processamento em lote
SUPPORTED_FORMATS = {".mp3", ".wav", ".flac", ".m4a"}
