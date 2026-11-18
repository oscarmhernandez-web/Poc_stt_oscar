"""
cleanup.py - Limpieza de recursos NO productivos del POC de despliegue canary.

NO toca produccion: NUNCA modifica el alias 'champion' ni su version,
y por defecto no borra el endpoint (delete_test_endpoint=False).

Que hace (con dry_run=True por defecto, solo imprime):
  1. Protege e informa cual es la version champion (no se toca).
  2. Quita el alias 'challenger' del modelo si existe.
  3. Re-aplica el tag status=archived a versiones no-champion ya marcadas archived.
  4. Opcionalmente elimina el endpoint de TEST (solo si delete_test_endpoint=True).

Uso (en un notebook Databricks):
    import sys; sys.path.insert(0, "/Workspace/Users/<tu-usuario>/audio_poc/deployment")
    from cleanup import cleanup
    cleanup("workspace.audio_poc.whistle_asr", "whistle-asr-endpoint",
            delete_test_endpoint=False, dry_run=True)   # dry_run=False para ejecutar
"""
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")


def cleanup(model_name, endpoint_name=None, delete_test_endpoint=False, dry_run=True):
    client = MlflowClient()

    # 1. identificar y proteger el champion
    champion_version = None
    try:
        champion_version = str(client.get_model_version_by_alias(model_name, "champion").version)
        print(f"[protect] champion = v{champion_version} (no se tocara)")
    except Exception:
        print("[protect] no hay alias champion; nada de produccion que proteger")

    # 2. quitar alias challenger (idempotente)
    try:
        client.delete_registered_model_alias(model_name, "challenger")
        print(f"[alias] challenger eliminado de {model_name}")
    except Exception:
        print("[alias] no habia alias challenger")

    # 3. (re)marcar como archived las versiones no-champion con tag status=archived
    for v in client.search_model_versions(f"name='{model_name}'"):
        ver = str(v.version)
        if ver == champion_version:
            continue  # NUNCA tocar champion
        tags = v.tags or {}
        if tags.get("status") == "archived":
            if dry_run:
                print(f"[dry-run] archivaria v{ver} (status=archived ya) -- informativo")
            else:
                client.set_model_version_tag(model_name, ver, "status", "archived")
                print(f"[archive] v{ver} tag status=archived")

    # 4. endpoint de test (opcional, nunca por defecto)
    if delete_test_endpoint and endpoint_name:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        if dry_run:
            print(f"[dry-run] eliminaria endpoint de test {endpoint_name}")
        else:
            try:
                w.serving_endpoints.delete(endpoint_name)
                print(f"[endpoint] endpoint de test {endpoint_name} eliminado")
            except Exception as e:
                print(f"[endpoint] no se pudo eliminar {endpoint_name}: {e}")

    print("cleanup completado (produccion intacta).")


if __name__ == "__main__":
    cleanup(
        model_name="workspace.audio_poc.whistle_asr",
        endpoint_name="whistle-asr-endpoint",
        delete_test_endpoint=False,
        dry_run=True,
    )
