# Music Cutter — Preparador de Faixas para Rádio

Ferramenta desktop para **processamento em lote** de arquivos de áudio (`.mp3`, `.wav`, `.flac`), preparando-os para automação de rádio com crossfade.

---

## Estrutura do Projeto

```
music-cutter/
├── app.py              # Interface gráfica (CustomTkinter) + orquestração
├── audio_processor.py  # Lógica de processamento de áudio (pydub)
├── requirements.txt    # Dependências Python
└── README.md
```

---

## Pré-requisitos

### 1. FFmpeg
O `pydub` usa o FFmpeg por trás. Instale e garanta que esteja no PATH do sistema.

- **Windows**: Baixe em https://ffmpeg.org/download.html → Adicione a pasta `bin/` ao PATH.
- **Linux**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`

### 2. Dependências Python

```bash
pip install customtkinter pydub
```

---

## Como Usar

```bash
python app.py
```

---

## Parâmetros

| Campo | Padrão | Descrição |
|---|---|---|
| Limiar de Corte — Início (dBFS) | `-20` | Volume mínimo para considerar que a música começa |
| Limiar de Corte — Fim (dBFS) | `-25` | Volume abaixo do qual o final da música é descartado |
| Fade-in no Início (ms) | `200` | Duração do fade-in aplicado ao início cortado |
| Fade-out no Fim (ms) | `2000` | Duração do fade-out aplicado ao fim cortado |
| Silêncio adicionado ao Final (ms) | `2000` | Silêncio absoluto concatenado ao fim (para crossfade) |

---

## Fluxo de Processamento por Arquivo

```
[Arquivo Original]
      ↓
 1. Carrega o áudio
      ↓
 2. Varre do início → encontra onde volume > Limiar Início → corta tudo antes
      ↓
 3. Varre do fim ← encontra onde volume > Limiar Fim → corta tudo depois
      ↓
 4. Aplica fade-in no início + fade-out no fim
      ↓
 5. Concatena silêncio absoluto ao final
      ↓
 6. Exporta com mesmo formato e bitrate original
```

---

## Notas Técnicas

- O processamento roda em **thread separada** (via `threading` + `queue.Queue`) para a interface nunca travar.
- Arquivos corrompidos ou inválidos são **ignorados com log de erro**, continuando o lote.
- A pasta de destino é **criada automaticamente** se não existir.
- Para MP3, o **bitrate original** é lido via `pydub.utils.mediainfo` e mantido na exportação.
