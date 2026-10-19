"""Gera deepfake_voz_ptbr_cnn.ipynb (Colab) a partir das células "# %%" do .py, que é a fonte única.

Uso:  python ferramentas/gerar_notebook.py            # (re)gera o notebook
      python ferramentas/gerar_notebook.py --checar   # sai com 1 se o notebook estiver desatualizado
"""
import json, re, sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PY, IPYNB = RAIZ / "deepfake_voz_ptbr_cnn.py", RAIZ / "deepfake_voz_ptbr_cnn.ipynb"


def celulas(texto):
    """Divide o .py nas linhas "# %%"; o cabeçalho antes da 1ª vira uma célula de código."""
    partes = re.split(r"^(# %%.*)\n", texto, flags=re.M)
    saida = [("code", partes[0])] if partes[0].strip() else []
    for marcador, corpo in zip(partes[1::2], partes[2::2]):
        if "[markdown]" in marcador:
            saida.append(("markdown", re.sub(r"^# ?", "", corpo.rstrip("\n"), flags=re.M)))
        else:
            saida.append(("code", corpo.rstrip("\n")))
    return saida


def notebook(texto):
    cells = []
    for i, (tipo, fonte) in enumerate(celulas(texto)):
        # ids fixos (exigidos pelo nbformat 4.5), para o editor não reescrever o arquivo
        c = {"cell_type": tipo, "id": f"celula-{i:02d}", "metadata": {}, "source": fonte.splitlines(keepends=True)}
        if tipo == "code":
            c.update(execution_count=None, outputs=[])
        cells.append(c)
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                         "kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"}}}


def serializar(nb):
    return json.dumps(nb, ensure_ascii=False, indent=1) + "\n"


if __name__ == "__main__":
    novo = serializar(notebook(PY.read_text(encoding="utf-8")))
    if "--checar" in sys.argv:
        ok = IPYNB.exists() and IPYNB.read_text(encoding="utf-8") == novo
        print("notebook em dia" if ok else f"{IPYNB.name} desatualizado: rode python ferramentas/gerar_notebook.py")
        sys.exit(0 if ok else 1)
    IPYNB.write_text(novo, encoding="utf-8")
    print(f"{IPYNB.name}: {len(json.loads(novo)['cells'])} células")
