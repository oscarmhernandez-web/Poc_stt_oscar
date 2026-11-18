# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,Descripción
# MAGIC %md
# MAGIC # 00 · Configuración y utilidades comunes
# MAGIC No se ejecuta solo: los demás notebooks lo cargan con `%run ./utils/00_config`.
# MAGIC Contiene rutas y parámetros, el motor ASR (`load_asr`, `transcribe`), la evaluación (`eval_batch`, `summarize`)
# MAGIC y su registro en MLflow (`run_eval`). No lleva `%pip` (no se puede reiniciar Python dentro de un `%run`).

# COMMAND ----------

# DBTITLE 1,Rutas, parámetros y ffmpeg
import os, glob, json, time, shutil, subprocess, tempfile
import jiwer, pandas as pd, soundfile as sf

# --- Rutas (Volume de Unity Catalog) ---
CATALOG, SCHEMA, VOLUME = "workspace", "audio_poc", "audio"
BASE = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
AUDIO_DIR = f"{BASE}/opus"        # audios de entrada (.opus / .ogg / .wav), anonimizados o sintéticos
REF_DIR   = f"{BASE}/ref_txt"     # <nombre>.txt con la transcripción correcta (opcional)
AZURE_DIR = f"{BASE}/azure_txt"   # <nombre>.txt con la salida de Azure STT (para comparar)

# --- Modelo ---
BACKEND  = "whistle"                 # "whistle" (Cactus, CPU) | "whisper" (transformers, baseline)
MODEL_ID = "Cactus-Compute/whistle"  # para whisper, p. ej. "openai/whisper-small"
LANGUAGE = "es"
UC_MODEL = f"{CATALOG}.{SCHEMA}.whistle_asr"

# --- Costes ---
CLUSTER_EUR_PER_HOUR = 1.0        # coste real del clúster (DBU + VM) en €/h
AZURE_EUR_PER_HOUR   = 0.151

# Si ffmpeg no está en el PATH, se usa el binario que trae imageio-ffmpeg (si está instalado).
if shutil.which("ffmpeg") is None:
    try:
        import imageio_ffmpeg
        _bin = tempfile.mkdtemp()
        os.symlink(imageio_ffmpeg.get_ffmpeg_exe(), f"{_bin}/ffmpeg")
        os.environ["PATH"] = f"{_bin}:{os.environ['PATH']}"
    except ImportError:
        print("Aviso: ffmpeg no encontrado. Instala imageio-ffmpeg o usa el init script install_ffmpeg.sh")

def audio_files():
    """Lista ordenada de audios (.opus, .ogg, .wav) en AUDIO_DIR."""
    return sorted(f for ext in ("opus", "ogg", "wav") for f in glob.glob(f"{AUDIO_DIR}/*.{ext}"))

# COMMAND ----------

# DBTITLE 1,Motor ASR: carga y transcripción
def to_wav16k(path):
    """Convierte cualquier audio a WAV 16 kHz mono (formato que exige Whistle). Devuelve la ruta en /tmp."""
    name = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(tempfile.gettempdir(), f"{name}_16k.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-ar", "16000", "-ac", "1", out], check=True)
    return out

def group_words(words, max_gap=0.8):
    """Agrupa palabras con tiempos en segmentos; una pausa > max_gap segundos abre un segmento nuevo."""
    segs, cur = [], []
    for w in words:
        if cur and w["start"] - cur[-1]["end"] > max_gap:
            segs.append(cur); cur = []
        cur.append(w)
    if cur: segs.append(cur)
    return [{"speaker": None, "start": s[0]["start"], "end": s[-1]["end"],
             "text": " ".join(x["word"] for x in s)} for s in segs]

def load_asr():
    """Devuelve el motor ASR según BACKEND: el módulo cactus (Whistle) o un pipeline de transformers (Whisper)."""
    if BACKEND == "whistle":
        import cactus
        subprocess.run(["cactus", "download", "whistle"], check=False)   # descarga el modelo si falta
        return cactus
    import torch
    from transformers import pipeline
    gpu = torch.cuda.is_available()
    return pipeline("automatic-speech-recognition", model=MODEL_ID, device=0 if gpu else -1,
                    torch_dtype=torch.float16 if gpu else torch.float32)

def transcribe(asr, path):
    """Devuelve {"text", "segments": [{speaker, start, end, text}]}. speaker=None (la diarización va aparte)."""
    if BACKEND == "whistle":
        # Siempre se convierte: hay .wav a 8 kHz / estéreo que Whistle no acepta.
        # OJO: verifica la forma de r["words"] en el README de cactus-compute.
        r = asr.transcribe(to_wav16k(path), language=LANGUAGE, word_timestamps=True)
        words = [{"word": w.get("word") or w.get("text"), "start": w["start"], "end": w["end"]}
                 for w in r.get("words", [])]
        return {"text": r["text"].strip(), "segments": group_words(words)}

    # Whisper: trozos de 30 s con timestamps por trozo
    out = asr(path, chunk_length_s=30, return_timestamps=True,
              generate_kwargs={"language": {"es": "spanish"}.get(LANGUAGE, LANGUAGE), "task": "transcribe"})
    segs = [{"speaker": None, "start": c["timestamp"][0], "end": c["timestamp"][1], "text": c["text"].strip()}
            for c in out.get("chunks", [])]
    return {"text": out["text"].strip(), "segments": segs}

# COMMAND ----------

# DBTITLE 1,Evaluación: WER, RTF y coste
# Normalización antes del WER: minúsculas, sin puntuación ni espacios sobrantes
NORM = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                      jiwer.RemoveMultipleSpaces(), jiwer.Strip()])

def read_ref(path):
    """Transcripción de referencia (REF_DIR/<nombre>.txt) o None. Ignora el sufijo _16k de to_wav16k."""
    name = os.path.splitext(os.path.basename(path))[0].removesuffix("_16k")
    p = f"{REF_DIR}/{name}.txt"
    return open(p, encoding="utf-8").read() if os.path.exists(p) else None

def eval_batch(asr, paths):
    """Transcribe cada audio. Devuelve un DataFrame con duración, tiempo, RTF, WER, hipótesis y referencia."""
    rows = []
    for p in paths:
        dur, t0 = sf.info(p).duration, time.time()
        hyp = transcribe(asr, p)["text"]
        el = time.time() - t0
        ref = read_ref(p)
        rows.append({"file": os.path.basename(p), "audio_s": dur, "proc_s": el, "rtf": el / dur,
                     "wer": jiwer.wer(NORM(ref), NORM(hyp)) if ref else None, "hyp": hyp, "ref": ref})
    return pd.DataFrame(rows)

def summarize(df):
    """Métricas globales: RTF, tiempo medio, coste por hora de audio y WER (solo si hay referencias)."""
    rtf = df.proc_s.sum() / df.audio_s.sum()
    m = {"mean_proc_s_per_call": df.proc_s.mean(), "rtf": rtf,
         "est_cost_eur_per_audio_hour": CLUSTER_EUR_PER_HOUR * rtf}
    r = df.dropna(subset=["ref"])
    if len(r):
        m["wer_mean"] = r.wer.mean()
        m["wer_corpus"] = jiwer.wer([NORM(x) for x in r.ref], [NORM(x) for x in r.hyp])
    return m

# COMMAND ----------

# DBTITLE 1,Registro en MLflow
from collections import Counter

# Experimento MLflow del usuario actual (mlflow se importa dentro de las funciones: 01 no lo necesita)
EXPERIMENT = f"/Users/{spark.sql('select current_user()').first()[0]}/asr_poc"

def frequent_errors(df, top=50):
    """Sustituciones y omisiones más frecuentes (ref -> hip), alineando palabra a palabra. No cuenta inserciones."""
    c = Counter()
    for _, r in df.dropna(subset=["ref"]).iterrows():
        ref_w, hyp_w = NORM(r.ref).split(), NORM(r.hyp).split()
        for ch in jiwer.process_words(NORM(r.ref), NORM(r.hyp)).alignments[0]:
            if ch.type == "substitute":
                c.update(f"{a}->{b}" for a, b in zip(ref_w[ch.ref_start_idx:ch.ref_end_idx],
                                                    hyp_w[ch.hyp_start_idx:ch.hyp_end_idx]))
            elif ch.type == "delete":
                c.update(f"{a}->(omitida)" for a in ref_w[ch.ref_start_idx:ch.ref_end_idx])
    return dict(c.most_common(top))

def run_eval(asr, paths, run_name, fmt="opus"):
    """eval_batch + summarize, registrando parámetros, métricas y transcripciones en un run de MLflow."""
    import mlflow, resource
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"backend": BACKEND, "model": MODEL_ID, "language": LANGUAGE,
                           "audio_format": fmt, "n_audios": len(paths)})
        df = eval_batch(asr, paths)
        m = summarize(df)
        m["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024   # RAM pico (Whistle va en CPU)
        if "wer_corpus" in m:
            mlflow.log_dict(frequent_errors(df), "frequent_errors.json")
        mlflow.log_metrics({k: float(v) for k, v in m.items()})
        mlflow.log_table(df, "transcriptions.json")
        print(m)
    return df, m