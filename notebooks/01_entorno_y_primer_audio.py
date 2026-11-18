# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,Introducción
# MAGIC %md
# MAGIC # 01 · Entorno y primer audio
# MAGIC Objetivo: comprobar que el entorno funciona y transcribir **un solo audio**. Nada de métricas todavía.
# MAGIC
# MAGIC Antes de empezar:
# MAGIC 1. Clúster ML Runtime (Whistle corre en CPU; GPU solo hace falta para el baseline Whisper).
# MAGIC 2. `ffmpeg`: si no está en el clúster, `00_config` usa el binario de `imageio-ffmpeg` (se instala abajo).
# MAGIC 3. Ajusta rutas y parámetros en `utils/00_config`.
# MAGIC 4. Sube 1-3 audios **sin datos personales** al Volume (`AUDIO_DIR`).

# COMMAND ----------

# DBTITLE 1,Instalar dependencias
# MAGIC %pip install -q cactus-compute transformers accelerate soundfile jiwer imageio-ffmpeg
# MAGIC # Paquetes:
# MAGIC #   cactus-compute -> SDK de Cactus para el modelo Whistle (ASR en CPU)
# MAGIC #   transformers / accelerate -> baseline Whisper de HuggingFace (BACKEND="whisper")
# MAGIC #   soundfile -> leer metadatos/contenido de audio (WAV, OGG, OPUS)
# MAGIC #   jiwer -> cálculo de WER (se usa en notebooks posteriores)
# MAGIC #   imageio-ffmpeg -> trae un binario de ffmpeg sin necesidad de apt/init script
# MAGIC
# MAGIC # Reinicia el intérprete de Python para que los paquetes recién instalados se puedan importar.
# MAGIC # OJO: borra todas las variables; por eso esta celda va la primera.
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Exponer ffmpeg en el PATH
# to_wav16k() (en 00_config) llama al comando "ffmpeg" por su nombre, así que debe estar en el PATH.
# imageio-ffmpeg descarga el binario con un nombre largo; aquí creamos un enlace llamado "ffmpeg"
# en una carpeta temporal y la anteponemos al PATH del proceso.
import imageio_ffmpeg, os, tempfile, subprocess
d = tempfile.mkdtemp()                                         # carpeta temporal para el enlace
os.symlink(imageio_ffmpeg.get_ffmpeg_exe(), d + '/ffmpeg')     # enlace simbólico "ffmpeg" -> binario real
os.environ['PATH'] = d + ':' + os.environ['PATH']              # hacerlo visible a subprocess

# Comprobación: imprime la primera línea de "ffmpeg -version"
print(subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True).stdout.splitlines()[0])

# COMMAND ----------

# DBTITLE 1,Cargar configuración común
# MAGIC %run ./utils/00_config

# COMMAND ----------

# DBTITLE 1,Preparar esquema, volumen y carpetas
# Crea catálogo/volumen si no existen (el catálogo y esquema deben existir ya o tener permisos para crearlos)
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
# Carpetas dentro del Volume (definidas en 00_config):
#   AUDIO_DIR -> audios de entrada, REF_DIR -> transcripciones correctas, AZURE_DIR -> salida de Azure STT
for d in (AUDIO_DIR, REF_DIR, AZURE_DIR):
    os.makedirs(d, exist_ok=True)   # exist_ok=True: no falla si ya existen

# Lista ordenada de audios (.opus/.ogg/.wav) en AUDIO_DIR
files = audio_files()
print(len(files), "audios en", AUDIO_DIR)

# COMMAND ----------

# DBTITLE 1,Comprobar que el audio se lee
# ¿Se puede leer el audio? (soundfile/libsndfile >= 1.1 lee OPUS/OGG)
import soundfile as sf
# sf.info solo lee la cabecera (no decodifica todo el audio): duración, frecuencia de muestreo y canales.
# Útil para ver si hará falta convertir: Whistle necesita 16 kHz mono (lo hace to_wav16k).
info = sf.info(files[0])
print(files[0], "-", info.duration, "s,", info.samplerate, "Hz,", info.channels, "canales")

# COMMAND ----------

# DBTITLE 1,Transcribe first audio
# Carga el motor ASR según BACKEND (00_config): módulo cactus (Whistle) o pipeline de Whisper
asr = load_asr()

t0 = time.time()  # medir el tiempo de transcripción (sirve para estimar coste vs Azure)
# transcribe convierte a WAV 16 kHz mono y devuelve {"text": ..., "segments": [{speaker, start, end, text}]}
res = transcribe(asr, files[0])
print(f"{time.time()-t0:.1f}s")

# Texto completo de la transcripción
print(res["text"])
# Primeros 5 segmentos (palabras agrupadas por pausas > 0.8 s); speaker=None hasta la diarización
print(json.dumps(res["segments"][:5], ensure_ascii=False, indent=2))

# COMMAND ----------

# DBTITLE 1,Siguiente paso
# MAGIC %md
# MAGIC **Listo si**: ves texto en español coherente y segmentos con `start`/`end`.
# MAGIC Si `segments` sale vacío, revisa la forma del resultado de Cactus (`r["words"]`) en `utils/00_config`.
# MAGIC Siguiente: `02_evaluacion`.