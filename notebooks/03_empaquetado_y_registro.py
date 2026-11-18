# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# DBTITLE 1,Introducción
# MAGIC %md
# MAGIC # 03 · Empaquetado (pyfunc) y registro en Unity Catalog
# MAGIC Entrada: ruta de audio. Salida: JSON speaker/start/end/text. Registro con alias `candidate`.
# MAGIC Hazlo solo cuando los resultados de `02_evaluacion` sean razonables.

# COMMAND ----------

# DBTITLE 1,Instalar dependencias
# MAGIC %pip install -q cactus-compute soundfile jiwer mlflow huggingface_hub pandas
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Cargar configuración común
# MAGIC %run ./utils/00_config

# COMMAND ----------

# DBTITLE 1,Definir pyfunc WhistleASR
import mlflow, pandas as pd
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from mlflow.tracking import MlflowClient
from huggingface_hub import snapshot_download

mlflow.set_registry_uri("databricks-uc")

# La pyfunc debe ser autocontenida (se ejecuta en serving sin 00_config), por eso repite
# to_wav16k y group_words como métodos estáticos.
class WhistleASR(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import cactus
        self.cactus = cactus
        self.language = context.model_config.get("language", "es")
        # TODO(verificar): cómo indicar a cactus la ruta de whistle.cact (context.artifacts["model_file"]).
        self.model_file = context.artifacts["model_file"]

    @staticmethod
    def _wav16k(path):
        import subprocess, tempfile, os
        out = os.path.join(tempfile.gettempdir(), os.path.splitext(os.path.basename(path))[0] + "_16k.wav")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-ar", "16000", "-ac", "1", out], check=True)
        return out

    @staticmethod
    def _group(words, max_gap=0.8):
        segs, cur = [], []
        for w in words:
            if cur and w["start"] - cur[-1]["end"] > max_gap:
                segs.append(cur); cur = []
            cur.append(w)
        if cur: segs.append(cur)
        return [{"speaker": None, "start": s[0]["start"], "end": s[-1]["end"],
                 "text": " ".join(x["word"] for x in s)} for s in segs]

    def predict(self, context, model_input, params=None):
        import json
        outs = []
        for p in model_input["audio_path"]:
            # siempre se convierte: hay .wav a 8 kHz / estéreo que Whistle no acepta
            r = self.cactus.transcribe(self._wav16k(p), language=self.language, word_timestamps=True)
            words = [{"word": w.get("word") or w.get("text"), "start": w["start"], "end": w["end"]}
                     for w in r.get("words", [])]
            outs.append(json.dumps({"text": r["text"].strip(), "segments": self._group(words)}, ensure_ascii=False))
        return outs

# COMMAND ----------

# DBTITLE 1,Package and register model
files = audio_files()
example = pd.DataFrame({"audio_path": [files[0] if files else f"{AUDIO_DIR}/ejemplo.opus"]})
signature = ModelSignature(inputs=Schema([ColSpec("string", "audio_path")]),
                           outputs=Schema([ColSpec("string")]))

model_dir = snapshot_download(MODEL_ID, local_dir=os.path.join(tempfile.gettempdir(), "whistle"))
cact = glob.glob(f"{model_dir}/**/*.cact", recursive=True)
assert cact, "No hay .cact en el repo de Hugging Face; usa `cactus download whistle` y apunta a ese fichero"

with mlflow.start_run(run_name="package_whistle"):
    mlflow.pyfunc.log_model(
        name="model",   # MLflow < 3: artifact_path="model"
        python_model=WhistleASR(),
        artifacts={"model_file": cact[0]},
        signature=signature,
        input_example=example,
        model_config={"language": LANGUAGE},
        pip_requirements=["mlflow", "cactus-compute", "soundfile"],  # + ffmpeg en el clúster/endpoint
        registered_model_name=UC_MODEL,
    )

client = MlflowClient()
ver = max(int(v.version) for v in client.search_model_versions(f"name='{UC_MODEL}'"))
client.set_registered_model_alias(UC_MODEL, "candidate", ver)
print(f"{UC_MODEL} v{ver} -> candidate")
# Tras validar frente a Azure: client.set_registered_model_alias(UC_MODEL, "champion", ver)

# COMMAND ----------

# DBTITLE 1,Load candidate model and test predict
loaded = mlflow.pyfunc.load_model(f"models:/{UC_MODEL}@candidate")
print(loaded.predict(example)[0][:800])

# COMMAND ----------

# MAGIC %md
# MAGIC **Serving (opcional)**: para la prueba basta un job batch sobre el Volume llamando a `loaded.predict`.
# MAGIC Un endpoint no hace falta para un POC (y Whistle va en CPU). **Pendiente**: diarización para `speaker`.