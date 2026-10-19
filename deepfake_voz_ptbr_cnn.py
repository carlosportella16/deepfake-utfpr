# -*- coding: utf-8 -*-
"""
Detecção de voz deepfake em Português (PT-BR) com Rede Neural Convolucional (CNN)
=================================================================================

Classificação binária: voz REAL (rótulo 0) x voz FALSA/sintética (rótulo 1).
Entrada: log-mel-espectrograma de janelas de 3 s (128 bandas x 301 quadros).
Modelo: ensemble de 2 CNNs residuais treinadas do zero (P(fake) = média das duas).
Bases públicas: MLAAD, CORAA-MUPE, Common Voice e vozes comerciais (ElevenLabs, Polly, Speechify, Hume,
Luvvoice, Kokoro, Microsoft edge-tts, Google gTTS); In-the-Wild só para avaliação.
Os áudios do grupo (./fake e ./original) são só o teste cego: nunca entram no treino.

No Colab: Ambiente de execução > Alterar tipo > GPU (T4). Cada bloco "# %%" é uma célula.
No Mac:   MPLBACKEND=Agg .venv312/bin/python deepfake_voz_ptbr_cnn.py
Só classificar áudios com o modelo já treinado: rode apenas a seção 9.
"""

# %% [markdown]
# # Detecção de voz deepfake PT-BR com CNN
# **Real (0) x Falso (1)** a partir de log-mel-espectrogramas.
# Execute as células em ordem. Ajuste a configuração na próxima célula.

# %%
import os
# ============ CONFIGURAÇÃO (ajuste aqui) ============
N_TREINO_POR_CLASSE = 4000     # total de treino = 2 x este valor (3000 = perfil mais rápido no Mac)
N_VAL_POR_CLASSE    = 400
N_TESTE_POR_CLASSE  = 400

SEMENTE      = 42
EPOCAS_MAX   = 80              # agenda cosseno; com 40 épocas o modelo chegava na última ainda melhorando
PACIENCIA    = 15              # épocas sem melhorar a val_loss antes de parar (EarlyStopping)
BATCH_SIZE   = 64
LR_PICO      = 1.5e-3          # taxa de aprendizado máxima (aquecimento de 1 época + decaimento cosseno)
P_CODEC      = 0.7             # prob. de passar a amostra por um codec sorteado (mp3/opus/aac), nas DUAS classes
P_DEGRADAR   = 0.5             # prob. de aplicar as outras degradações de canal (telefone, ruído, eco...)
P_WHATSAPP   = 0.25            # dentro do codec: prob. de ser a cadeia de REENVIO (Opus -> MP3/AAC 24 kHz)
N_VERSOES_TREINO = 3           # versões de cada áudio de treino (recorte, codec e degradação próprios); 1 época = 1 sorteada

# ---- Fontes de dados (seção 2) ----
USAR_MLAAD = True              # +geradores TTS em português (exige login no HF e aceitar os termos)
USAR_COMERCIAIS = True         # vozes de serviços COMERCIAIS (Microsoft edge-tts, Google gTTS, ElevenLabs público)
USAR_CORAA = True              # vozes reais PT-BR de entrevistas (sotaques e gravações variadas)
USAR_COMMON_VOICE = True       # vozes reais PT gravadas por voluntários em celular/notebook (milhares de microfones)
FRAC_COMERCIAL = 0.5           # fração dos FALSOS que vem de serviços comerciais (o resto vem do MLAAD)
FRAC_CORAA = 0.4               # fração dos REAIS que vem do CORAA (o resto vem do Common Voice)
MAX_SHARDS_CORAA = 12
MAX_SHARDS_CV    = 8
# Geradores do MLAAD que NUNCA entram no treino: medem a generalização para TTS desconhecidos
MLAAD_GERADORES_TESTE = ["OpenAI TTS-1 HD", "Voxtral", "Qwen3-TTS-12Hz-1.7B-Base"]
N_GENERALIZACAO_POR_CLASSE = 200
# Falsos do MLAAD em OUTROS idiomas: trazem geradores que o pt não tem (ElevenLabs v2/v3/Turbo, Cartesia, MiniMax,
# Gemini, Bark...). O detector aprende o som do vocoder, não o idioma; e os reais vêm dos MESMOS idiomas na mesma
# fração (Common Voice), para "idioma estrangeiro = falso" não virar atalho.
MLAAD_IDIOMAS_EXTRA = ["en", "es", "de", "fr", "it"]
FRAC_MLAAD_OUTROS_IDIOMAS = 0.2  # fração dos FALSOS (dividida igualmente entre os idiomas)
# Famílias reservadas à generalização: ficam fora do treino em QUALQUER idioma extra (ex.: outra versão do Qwen3-TTS)
MLAAD_FAMILIAS_RESERVADAS = ["openai", "voxtral", "qwen3-tts"]
REPASSAR_VAGAS = True          # vaga que um gerador não preenche passa para os que ainda têm clipes
PESOS_GERADORES = {"bark": 2.0}  # geradores (por trecho do nome) com cota maior: o Bark era o menos detectado (44%)
# Avaliação extra, FORA do treino: falsos e reais de figuras públicas tirados da internet (inglês)
USAR_IN_THE_WILD = True
N_IN_THE_WILD_POR_CLASSE = 400
# Quantos áudios comerciais gerar por script (uma vez só; ficam em cache no disco)
N_GERAR_EDGE_TTS = 2500
N_GERAR_GTTS     = 200

# ---- Desempenho do treino ----
USAR_PRECISAO_MISTA = True     # float16 nas convoluções; só liga com GPU NVIDIA (CUDA). No Mac (Metal) fica float32
USAR_XLA            = True     # compila o passo de treino; só com CUDA (no Metal o jit_compile é instável)
LIMITE_DS_NA_RAM_GB = 3.0      # até este tamanho o treino monta os lotes no tf.data (sem laço Python)
N_TRABALHADORES     = max(1, (os.cpu_count() or 2) - 2)   # threads para extrair características em paralelo
# ---- Ensemble: N modelos com sementes diferentes; P(fake) = média deles (menos variância, menos excesso de confiança)
N_MODELOS_ENSEMBLE = 2       # 2 residuais: EER degradado 4,9% x 5,4% de 1 modelo, teste cego igual (README, decisão 25)

# ---- Seção 4: comparação de ativações e otimizadores ----
COMPARAR_CONFIGURACOES = False # já rodado: ReLU + Adam venceu (True para repetir, ≈50 min)
EPOCAS_COMPARACAO      = 12

# ---- Contra "atalhos" ----
IGUALAR_PISO_DE_RUIDO = True   # falsos recebem ruído de fundo parecido com o dos reais (evita 'limpo = fake')
P_IGUALAR_PISO   = 0.7

# ---- Decisão ----
LIMIAR = 0.5                   # usado só nas métricas padrão (acurácia@0,5, F1...)
# Limiar de operação = ponto de EER na validação (erro em reais = erro em falsos: equilibrado).
MARGEM_INCERTA = 0.05          # P(fake) a menos de 0,05 do limiar de operação -> "INCONCLUSIVO"


# %%
# !pip install -q datasets librosa soundfile scikit-learn seaborn tensorflow huggingface_hub
# (no Colab, descomente a linha acima — ou rode a célula seguinte)
try:
    get_ipython().system("pip install -q datasets librosa soundfile scikit-learn seaborn huggingface_hub edge-tts gTTS")  # noqa
except NameError:
    pass

# %%
import os, gc, time, json, subprocess, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import scipy.signal as sps
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

np.random.seed(SEMENTE)
tf.keras.utils.set_random_seed(SEMENTE)
print("TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus if gpus else "nenhuma: treino na CPU (no Mac: instale tensorflow-metal; no Colab: ative a GPU)")
# Precisão mista e XLA só com GPU NVIDIA: na GPU Metal do Mac o float16 e o jit_compile são instáveis
CUDA = bool(gpus) and tf.test.is_built_with_cuda()
if USAR_PRECISAO_MISTA and CUDA:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")   # saída do modelo continua em float32
print("Política de precisão:", tf.keras.mixed_precision.global_policy().name)

def reiniciar_sessao(semente=SEMENTE, politica=None):
    """clear_session() do Keras também volta a política de precisão para float32: reaplica a escolhida."""
    politica = politica or tf.keras.mixed_precision.global_policy().name
    tf.keras.backend.clear_session()
    tf.keras.mixed_precision.set_global_policy(politica)
    tf.keras.utils.set_random_seed(semente)
# Perfil "GPU NVIDIA" (Colab T4): lote maior, LR escalado pela raiz do lote e mais threads de extração
# (o ffmpeg roda em processos próprios, então threads além do nº de núcleos ainda ajudam)
if CUDA:
    LR_PICO *= (128 / BATCH_SIZE) ** 0.5
    BATCH_SIZE = 128
    N_TRABALHADORES = max(N_TRABALHADORES, 2 * (os.cpu_count() or 2))
    print(f"Perfil CUDA: lote {BATCH_SIZE}, LR de pico {LR_PICO:.2e}, {N_TRABALHADORES} trabalhadores")
TEMPOS = {}                    # tempo de cada etapa (vai para o resultados.json)
assert shutil.which("ffmpeg"), "ffmpeg não encontrado (no Colab ele já vem instalado)."

# %%
# Google Drive: cache dos dados, modelos e resultados (a sessão do Colab pode cair)
try:
    from google.colab import drive
    drive.mount("/content/drive")
    PASTA_PROJETO = "/content/drive/MyDrive/projeto_deepfake_audio"
except ImportError:
    PASTA_PROJETO = "./projeto_deepfake_audio"
os.makedirs(PASTA_PROJETO, exist_ok=True)

print("Pasta do projeto:", PASTA_PROJETO)

# %% [markdown]
# ## 1. Pré-processamento do áudio
# Mono → 16 kHz → remove silêncio das pontas → normaliza pico → recorte de 3 s →
# (treino) codec e degradação de canal aleatórios → log-mel normalizado.
#
# Por que continuar em 16 kHz? O corte natural de banda já muda com o formato (Opus do WhatsApp corta
# em ~12 kHz, ElevenLabs em ~15 kHz). Acima de 8 kHz a CNN aprenderia "largura de banda = classe".
# Em 16 kHz todos os formatos cobrem 0–8 kHz por inteiro.

# %%
SR        = 16000
DURACAO   = 3
N_AMOSTRAS = SR * DURACAO
PASSO_JANELA = SR * 3 // 2             # janelas de 3 s com passo de 1,5 s
HOP       = 160                        # 10 ms
N_FRAMES  = 1 + N_AMOSTRAS // HOP      # 301
DUR_MIN   = 1.0                        # descarta clipes com menos de 1 s de fala
N_MELS    = 128
N_FFT     = 1024
FORMA     = (N_MELS, N_FRAMES, 1)          # entrada da CNN

def normalizar(y):
    return (0.95 * y / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)

def preparar_onda(y, sr_original):
    """Mono, 16 kHz, sem silêncio nas pontas, pico normalizado."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=0) if y.shape[0] < y.shape[-1] else y.mean(axis=1)
    if sr_original != SR:
        y = librosa.resample(y, orig_sr=sr_original, target_sr=SR)
    y, _ = librosa.effects.trim(y, top_db=35)
    return normalizar(y)

def completar(y, n=N_AMOSTRAS):
    """Se curto, repete o próprio áudio (em vez de preencher com silêncio)."""
    if len(y) >= n:
        return y[:n]
    return np.tile(y, int(np.ceil(n / max(1, len(y)))))[:n]

def recorte_aleatorio(y, rng, n=N_AMOSTRAS):
    if len(y) <= n:
        return completar(y, n)
    ini = rng.integers(0, len(y) - n + 1)
    return y[ini:ini + n]

def janelas(y, n=N_AMOSTRAS, passo=PASSO_JANELA):
    """Janelas deslizantes de 3 s (passo 1,5 s) cobrindo o áudio inteiro."""
    if len(y) <= n:
        return [completar(y, n)]
    inicios = list(range(0, len(y) - n + 1, passo))
    if inicios[-1] != len(y) - n:
        inicios.append(len(y) - n)
    return [y[i:i + n] for i in inicios]

def onda_para_mel(y):
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS, fmin=20, fmax=SR // 2)
    S_db = librosa.power_to_db(S, ref=np.max, top_db=80.0)      # [-80, 0] dB
    return ((S_db + 80.0) / 80.0).astype(np.float32)             # [0, 1]

def caracteristicas(y):
    """Onda de 3 s -> (128 bandas, 301 quadros, 1 canal)."""
    return onda_para_mel(y)[..., None]

# --------- degradações de canal (aplicadas às DUAS classes, igualmente) ---------
def _ffmpeg(args, dados):
    return subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args],
                          input=dados, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=True).stdout

def aplicar_codec(y, codec="mp3", kbps=64, sr_codec=None):
    """Comprime e descomprime via ffmpeg: MP3, Opus (WhatsApp) ou AAC (m4a de celular, Instagram, iPhone).
    sr_codec: taxa de amostragem dentro do arquivo comprimido (padrão: a do projeto)."""
    entrada = ["-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0"] + (["-ar", str(sr_codec)] if sr_codec else [])
    if codec == "mp3":
        saida = ["-c:a", "libmp3lame", "-b:a", f"{kbps}k", "-f", "mp3", "pipe:1"]
    elif codec == "aac":
        saida = ["-c:a", "aac", "-b:a", f"{kbps}k", "-f", "adts", "pipe:1"]
    else:
        saida = ["-c:a", "libopus", "-b:a", f"{kbps}k", "-f", "ogg", "pipe:1"]
    comprimido = _ffmpeg(entrada + saida, y.astype(np.float32).tobytes())
    dec = _ffmpeg(["-i", "pipe:0", "-f", "f32le", "-ar", str(SR), "-ac", "1", "pipe:1"], comprimido)
    return librosa.util.fix_length(np.frombuffer(dec, dtype=np.float32).copy(), size=len(y))

def _codec_ok(codec):
    try:
        aplicar_codec(np.random.randn(SR).astype(np.float32) * 0.1, codec, 32)
        return True
    except Exception:
        return False

CODECS = [c for c in ("mp3", "opus", "aac") if _codec_ok(c)]
print("Codecs disponíveis para aumento de dados:", CODECS)
TAXAS_CODEC = {"mp3": [24, 32, 48, 64, 96, 128], "opus": [12, 16, 24, 32], "aac": [32, 48, 64, 96, 128]}

def deg_codec(y, rng):
    codec = str(rng.choice(CODECS))
    return aplicar_codec(y, codec, int(rng.choice(TAXAS_CODEC[codec])))

def deg_whatsapp(y, rng):
    """Cadeia de REENVIO: nota de voz em Opus de baixa taxa, depois baixada e recodificada (MP3/AAC a 24 kHz,
    como nos .mpeg reenviados) e normalizada pelo pico. Nas DUAS classes, com a mesma probabilidade."""
    y = aplicar_codec(y, "opus", int(rng.choice([16, 24, 32])))
    codec = str(rng.choice([c for c in ("mp3", "aac") if c in CODECS] or ["opus"]))
    return aplicar_codec(y, codec, int(rng.choice([32, 48, 64])), sr_codec=24000 if codec != "opus" else None)

def deg_equalizacao(y, rng):
    """Resposta de frequência aleatória (±6 dB, suave): imita microfones diferentes (celular, headset, notebook)."""
    freqs = np.array([0, 150, 400, 1000, 2500, 5000, SR / 2]) / (SR / 2)
    ganho_db = rng.uniform(-6, 6, size=len(freqs))
    ganho_db[0] = ganho_db[1] - rng.uniform(0, 12)                 # graves cortados (mic pequeno)
    fir = sps.firwin2(129, freqs, 10 ** (ganho_db / 20))
    return sps.fftconvolve(y, fir, mode="same").astype(np.float32)

def deg_telefone(y, rng):
    z = librosa.resample(y, orig_sr=SR, target_sr=8000)
    z = sps.sosfilt(sps.butter(4, [300, 3400], btype="band", fs=8000, output="sos"), z)
    return librosa.util.fix_length(librosa.resample(z, orig_sr=8000, target_sr=SR), size=len(y))

def deg_ruido(y, rng):
    snr_db = rng.uniform(10, 35)
    p_sinal = np.mean(y ** 2) + 1e-12
    return y + rng.standard_normal(len(y)).astype(np.float32) * np.sqrt(p_sinal / 10 ** (snr_db / 10))

def deg_reverb(y, rng):
    rt60 = rng.uniform(0.15, 0.6)
    t = np.arange(int(SR * rt60)) / SR
    ri = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60)
    ri[0] = 3.0                                          # caminho direto
    molhado = sps.fftconvolve(y, ri)[:len(y)]
    molhado *= np.max(np.abs(y)) / (np.max(np.abs(molhado)) + 1e-9)
    mix = rng.uniform(0.2, 0.6)
    return (1 - mix) * y + mix * molhado

def deg_passa_baixa(y, rng):
    fc = rng.uniform(3500, 7500)
    return sps.sosfilt(sps.butter(6, fc, btype="low", fs=SR, output="sos"), y)

DEGRADACOES = [deg_telefone, deg_ruido, deg_reverb, deg_passa_baixa, deg_equalizacao]

def degradar(y, rng):
    """Aplica 1 ou 2 degradações aleatórias (ordem aleatória), sem codec."""
    k = int(rng.integers(1, 3))
    for i in rng.choice(len(DEGRADACOES), size=k, replace=False):
        y = DEGRADACOES[i](y, rng)
    return normalizar(np.asarray(y, dtype=np.float32))

def canal_aleatorio(y, rng, p_codec=P_CODEC, p_degradar=P_DEGRADAR, p_whatsapp=None):
    """Canal sorteado SEM olhar o rótulo: as duas classes recebem codec e degradações na mesma proporção.
    Devolve (onda, canal): canal = "" (sem codec), "codec" ou "whatsapp" (cadeia de reenvio)."""
    p_whatsapp = P_WHATSAPP if p_whatsapp is None else p_whatsapp
    if rng.random() < p_degradar:
        y = degradar(y, rng)
    canal = ""
    if CODECS and rng.random() < p_codec:
        # (com p_whatsapp = 0 não sorteia nada a mais: reproduz exatamente os sorteios da versão anterior)
        canal = "whatsapp" if "opus" in CODECS and p_whatsapp > 0 and rng.random() < p_whatsapp else "codec"
        y = normalizar(deg_whatsapp(y, rng) if canal == "whatsapp" else deg_codec(y, rng))
    return y, canal

# --------- anti-atalho: PISO DE RUÍDO ---------
# Voz sintética tem silêncio "digital" entre as palavras; gravação real sempre tem algum ruído de fundo.
# Sem isso, a CNN aprende "tem ruído = real, limpo = falso" (e uma gravação real feita num lugar silencioso
# com um bom microfone vira FALSA). No treino, parte dos falsos recebe o fundo de um real sorteado.
def piso_ruido_db(y):
    """Nível do 'fundo' do áudio: percentil 10 da energia dos quadros de 25 ms (em dB)."""
    r = librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0]
    return float(np.percentile(20 * np.log10(r + 1e-7), 10))

def igualar_ruido(y, alvo_db, rng):
    """Se o trecho é mais limpo que o alvo, soma ruído (branco ou rosa) até o fundo atingir o alvo."""
    atual = piso_ruido_db(y)
    if alvo_db <= atual + 1.0:
        return y
    var = 10 ** (alvo_db / 10) - 10 ** (atual / 10)
    ruido = rng.standard_normal(len(y)).astype(np.float32)
    if rng.random() < 0.5:
        ruido = sps.lfilter([1.0], [1.0, -0.9], ruido).astype(np.float32)      # ruído "rosa" (mais grave)
    ruido *= np.sqrt(var) / (ruido.std() + 1e-9)
    return normalizar(y + ruido)

# teste rápido
_rng = np.random.default_rng(0)
_y = normalizar(np.random.randn(SR * 5).astype(np.float32))
for f in DEGRADACOES + ([deg_codec] if CODECS else []):
    assert f(_y[:N_AMOSTRAS], _rng).shape == (N_AMOSTRAS,), f.__name__
assert onda_para_mel(recorte_aleatorio(_y, _rng)).shape == (N_MELS, N_FRAMES)
assert len(janelas(np.zeros(18 * SR, np.float32))) == 11   # 18 s -> 11 janelas (0; 1,5; ...; 15 s)

# %% [markdown]
# ## 2. Coleta dos dados (fontes públicas, nenhuma gravação do grupo)
# - **Comerciais**: Microsoft edge-tts e Google gTTS (pt, gerados uma vez) + ElevenLabs, Polly, Speechify, Hume,
#   Luvvoice e Kokoro (dataset público, inglês, com reais do YouTube em inglês para o idioma não virar pista).
# - **MLAAD**: só falsos, ~16 TTS em português e geradores extras em en/es/de/fr/it.
#   3 geradores ficam FORA do treino para medir a generalização a TTS nunca vistos.
# - **CORAA-MUPE**: só reais, entrevistas em PT-BR com sotaques e gravações variadas.
# - **Common Voice (pt)**: só reais, frases lidas por milhares de voluntários no próprio celular ou
#   notebook. É a fonte mais parecida com "uma pessoa gravando a própria voz": é ela que ensina
#   ao modelo que microfone simples, eco de quarto e m4a/MP3 também são voz REAL.
# - **In-the-Wild**: figuras públicas tiradas da internet (inglês), só para avaliação.
#
# **Antes de rodar:** aceite os termos em https://huggingface.co/datasets/mueller91/MLAAD e
# cadastre o token em Secrets do Colab com o nome `HF_TOKEN`.

# %%
import io, hashlib, threading, functools
import concurrent.futures as cf
import soundfile as sf
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, list_repo_files, get_token

# ---- Login no Hugging Face (o MLAAD exige aceitar os termos) ----
# 1) Logado no HF, abra https://huggingface.co/datasets/mueller91/MLAAD e clique em "Agree/Access".
# 2) Crie um token "Read" em https://huggingface.co/settings/tokens (da MESMA conta).
# 3) No Colab: ícone de chave (Secrets) -> HF_TOKEN = seu token, com "Acesso do notebook" ligado.
if get_token() is None:
    try:
        from google.colab import userdata
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    except Exception:
        from huggingface_hub import interpreter_login
        interpreter_login()                     # pede o token numa caixa de texto
print("Token do Hugging Face encontrado:", get_token() is not None)

FS = HfFileSystem()
MLAAD  = "mueller91/MLAAD"                      # TTS em português (só falsos)
CORAA  = "nilc-nlp/CORAA-MUPE-ASR"              # entrevistas reais PT-BR (só reais)
CV     = "fixie-ai/common_voice_17_0"           # Common Voice 17 (CC0), cópia em parquet; pasta pt/

def decodificar_bytes(b):
    """Bytes de áudio (wav/flac/mp3...) -> (onda mono float32, taxa)."""
    try:
        y, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
        return (y.mean(axis=1) if y.ndim > 1 else y), int(sr)
    except Exception:
        bruto = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
                                "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
                               input=b, stdout=subprocess.PIPE, check=True).stdout
        return np.frombuffer(bruto, dtype=np.float32).copy(), SR

def _onda_valida(b):
    y, sr = decodificar_bytes(b)
    y = preparar_onda(y, sr)
    return y if len(y) >= DUR_MIN * SR else None

# IMPORTANTE (memória): os coletores NÃO guardam o áudio bruto. Cada áudio é convertido na hora
# pela função `proc` (recorte de 4 s -> mel-espectrograma float16, ~100 KB) e descartado.

# ---------------- leitor de parquet em pedaços pequenos (CORAA e Common Voice) ----------------
def abrir_parquet(repo, caminho):
    return FS.open(f"datasets/{repo}/{caminho}", "rb", block_size=8 * 2**20)

def itens_de_um_shard(repo, caminho, n, rng, rotulo_padrao, proc):
    """Lê linhas aleatórias do parquet em lotes de 16 (pouca RAM) e para assim que tem n."""
    itens, geradores = [], []
    with abrir_parquet(repo, caminho) as f:
        pf = pq.ParquetFile(f, pre_buffer=False, buffer_size=8 * 2**20)
        colunas = [c for c in ("audio", "model") if c in pf.schema_arrow.names]
        for rg in rng.permutation(pf.num_row_groups):
            n_linhas = pf.metadata.row_group(int(rg)).num_rows
            faltam = n - len(itens)
            quero = set(rng.choice(n_linhas, size=min(n_linhas, 2 * faltam + 5), replace=False).tolist())
            pos = 0
            for lote in pf.iter_batches(batch_size=16, row_groups=[int(rg)], columns=colunas):
                for j, linha in enumerate(lote.to_pylist()):
                    if pos + j not in quero or len(itens) >= n:
                        continue
                    try:
                        a = linha["audio"]
                        y = _onda_valida(a["bytes"] if isinstance(a, dict) else a)
                    except Exception as e:
                        print("  erro em amostra:", e); continue
                    if y is not None:
                        itens.append(proc(y)); geradores.append(str(linha.get("model") or rotulo_padrao))
                pos += lote.num_rows
                del lote
                if len(itens) >= n or pos > max(quero):
                    break
            if len(itens) >= n:
                break
    return itens, geradores

def coletar_parquet(repo, shards, n_total, rng, rotulo_padrao, max_shards, proc):
    if n_total <= 0:
        return [], []
    shards = list(rng.permutation(shards)[:max_shards])
    por_shard = int(np.ceil(n_total / len(shards)))
    itens, geradores, t0 = [], [], time.time()
    for i, sh in enumerate(shards):
        o, g = itens_de_um_shard(repo, sh, min(por_shard, n_total - len(itens)), rng, rotulo_padrao, proc)
        itens += o; geradores += g
        print(f"  [{repo.split('/')[-1]} {sh}] {i+1}/{len(shards)} -> {len(itens)} ({time.time()-t0:.0f}s)")
        gc.collect()
        if len(itens) >= n_total:
            break
    return itens[:n_total], geradores[:n_total]

def coletar_coraa(split, n, rng, proc):
    shards = sorted(f for f in list_repo_files(CORAA, repo_type="dataset")
                    if f.startswith(f"data/{split}-") and f.endswith(".parquet"))
    return coletar_parquet(CORAA, shards, n, rng, "real_coraa", MAX_SHARDS_CORAA, proc)

def _e_do_split(caminho, split):
    """Aceita pt/train-0000.parquet, pt/train/0000.parquet, pt/train_0.parquet..."""
    return any(p == split or p.startswith((f"{split}-", f"{split}_", f"{split}."))
               for p in caminho.split("/")[1:])

ARQUIVOS_CV = None

def coletar_common_voice(split, n, rng, proc, idioma="pt"):
    global ARQUIVOS_CV
    ARQUIVOS_CV = ARQUIVOS_CV or list_repo_files(CV, repo_type="dataset")
    shards = sorted(f for f in ARQUIVOS_CV
                    if f.startswith(f"{idioma}/") and f.endswith(".parquet") and _e_do_split(f, split))
    if not shards:
        raise FileNotFoundError(f"nenhum parquet {idioma}/{split} em {CV}")
    return coletar_parquet(CV, shards, n, rng, "real_commonvoice" if idioma == "pt" else f"real_cv_{idioma}",
                           MAX_SHARDS_CV, proc)

# ---------------- MLAAD: arquivos .wav por gerador ----------------
def _parte_mlaad(caminho):
    """Divide os arquivos de cada gerador de forma fixa: 70% treino, 10% validação, 20% teste.
    O sorteio usa só 'gerador/arquivo': a divisão é a mesma no Mac e no Colab."""
    h = int(hashlib.md5("/".join(caminho.split("/")[-2:]).encode()).hexdigest(), 16) % 10
    return "train" if h < 7 else ("validation" if h < 8 else "test")

def mlaad_geradores():
    return sorted(p.rstrip("/").split("/")[-1] for p in FS.ls(f"datasets/{MLAAD}/fake/pt", detail=False))

@functools.lru_cache(maxsize=None)
def _arquivos_mlaad(g):
    """g = "Gerador" (português) ou "idioma/Gerador" (outros idiomas)."""
    caminho = f"fake/{g}" if "/" in g else f"fake/pt/{g}"
    return sorted(p for p in FS.ls(f"datasets/{MLAAD}/{caminho}", detail=False) if p.endswith(".wav"))

def mlaad_geradores_extra():
    """{idioma: [idioma/Gerador, ...]} só com geradores que NÃO existem no pt e fora das famílias reservadas."""
    pt = set(MLAAD_GERADORES)
    saida = {}
    for idioma in MLAAD_IDIOMAS_EXTRA:
        try:
            gens = sorted(p.rstrip("/").split("/")[-1] for p in FS.ls(f"datasets/{MLAAD}/fake/{idioma}", detail=False))
        except Exception as e:
            print(f"⚠️ MLAAD {idioma} indisponível ({type(e).__name__}).")
            continue
        saida[idioma] = [f"{idioma}/{g}" for g in gens if g not in pt
                         and not any(f in g.lower() for f in MLAAD_FAMILIAS_RESERVADAS)]
    return saida

def _ler_bytes(caminho):
    try:
        with FS.open(caminho, "rb") as f:
            return f.read()
    except Exception:
        return None

def testar_acesso_mlaad(geradores):
    """Baixa 1 arquivo: listar funciona sem permissão, mas baixar exige ter aceitado os termos."""
    try:
        with FS.open(_arquivos_mlaad(geradores[0])[0], "rb") as f:
            f.read(1024)
        return True
    except Exception as e:
        msg = str(e).split("\n")[0][:150]
        print(f"⚠️ MLAAD: sem permissão para baixar ({msg}).\n"
              "   Faça login no HF, aceite os termos em https://huggingface.co/datasets/mueller91/MLAAD\n"
              "   e confira se o HF_TOKEN dos Secrets é da MESMA conta. Seguindo sem o MLAAD.")
        return False

def _peso(g):
    return next((w for trecho, w in PESOS_GERADORES.items() if trecho in g.lower()), 1.0)

def cotas_por_peso(n_total, geradores):
    """Divide n_total vagas pelos pesos (maior resto): a soma é exatamente n_total, sem corte aleatório depois.
    A mesma regra no plano de idiomas e na coleta: com poucos áudios, cotas arredondadas para cima e um corte
    aleatório no fim desequilibravam o idioma entre as classes."""
    gens = sorted(geradores)
    w = sum(_peso(g) for g in gens)
    exato = {g: n_total * _peso(g) / w for g in gens}
    cotas = {g: int(np.floor(v)) for g, v in exato.items()}
    for g in sorted(gens, key=lambda g: (cotas[g] - exato[g], g))[:n_total - sum(cotas.values())]:
        cotas[g] += 1
    return cotas

def repartir_vagas(deficit, sobras):
    """Divide `deficit` vagas entre os geradores com sobra, proporcional à sobra, sem passar da sobra de cada um."""
    extra, sobras = {}, {g: s for g, s in sobras.items() if s > 0}
    while deficit > 0 and sobras:
        tot = sum(sobras.values())
        for g in sorted(sobras):
            q = min(sobras[g], int(np.ceil(deficit * sobras[g] / tot)), deficit)
            extra[g] = extra.get(g, 0) + q; sobras[g] -= q; deficit -= q
        sobras = {g: s for g, s in sobras.items() if s > 0}
    return extra

def planejar_vagas(n_total, disponiveis):
    """Quantos clipes cada gerador entrega (mesma regra da coleta, sem ler nada): usado para planejar o idioma."""
    if n_total <= 0 or not disponiveis:
        return {}
    cotas = cotas_por_peso(n_total, disponiveis)
    obtidos = {g: min(d, cotas[g]) for g, d in disponiveis.items()}
    if REPASSAR_VAGAS:
        for g, q in repartir_vagas(n_total - sum(obtidos.values()),
                                   {g: d - obtidos[g] for g, d in disponiveis.items()}).items():
            obtidos[g] += q
    return obtidos

def coletar_por_gerador(split, n_total, rng, geradores, listar, ler, prefixo, proc, todas_as_partes=False):
    """1ª passada: cota de cada gerador = n_total x peso / soma dos pesos (leitura em paralelo).
    2ª passada (REPASSAR_VAGAS): o que faltou vai para os geradores que ainda têm clipes não usados.
    `listar(g)` devolve os caminhos do gerador g; `ler(caminho)` devolve os bytes (ou None)."""
    if n_total <= 0 or not geradores:
        return [], []
    itens, nomes, t0 = [], [], time.time()
    restantes, obtidos, cotas = {}, {}, cotas_por_peso(n_total, geradores)

    def ler_ate(g, candidatos, k):
        n_ok = n_falhas = 0
        with cf.ThreadPoolExecutor(16) as ex:
            for b in ex.map(ler, candidatos):
                if b is None:
                    n_falhas += 1; continue
                if n_ok >= k:
                    continue
                try:
                    y = _onda_valida(b)
                except Exception:
                    y = None
                if y is not None:
                    itens.append(proc(y)); nomes.append(f"{prefixo}:{g}"); n_ok += 1
        return n_ok, n_falhas

    for g in geradores:
        arqs = listar(g)
        if not todas_as_partes:
            arqs = [p for p in arqs if _parte_mlaad(p) == split]
        perm = list(rng.permutation(arqs))
        corte = int(cotas[g] * 1.3) + 2                              # folga p/ descartes
        n_ok, n_falhas = ler_ate(g, perm[:corte], cotas[g])
        obtidos[g], restantes[g] = n_ok, perm[corte:]
        print(f"  [{prefixo} {split}] {g}: {n_ok}/{cotas[g]} ok, {n_falhas} falhas ({time.time()-t0:.0f}s)")
    deficit = n_total - min(n_total, len(itens))
    if REPASSAR_VAGAS and deficit > 0:
        extra = repartir_vagas(deficit, {g: len(r) for g, r in restantes.items()})
        for g, q in extra.items():
            n_ok, _ = ler_ate(g, restantes[g][:int(q * 1.3) + 2], q)
            obtidos[g] += n_ok
        print(f"  [{prefixo} {split}] repasse de {deficit} vagas -> " +
              ", ".join(f"{g} +{q}" for g, q in extra.items() if q) + f" (total {len(itens)}/{n_total})")
    ordem = rng.permutation(len(itens))[:n_total]
    return [itens[i] for i in ordem], [nomes[i] for i in ordem]

def coletar_mlaad(split, n_total, rng, geradores, proc, todas_as_partes=False):
    return coletar_por_gerador(split, n_total, rng, geradores, _arquivos_mlaad, _ler_bytes, "mlaad", proc,
                               todas_as_partes)

# ---------------- Serviços COMERCIAIS (o tipo de voz que o modelo antigo não reconhecia) ----------------
# Gerados por script uma única vez e guardados em disco (as próximas execuções só leem os arquivos):
#   edge_tts/      vozes neurais da Microsoft (pt-BR e pt-PT), lendo frases do Common Voice
#   gtts/          Google Translate TTS (pt-BR e pt-PT)
#   hf_elevenlabs/, hf_polly/, hf_speechify/, hf_hume/, hf_luvvoice/, hf_kokoro/
#                  6 serviços comerciais em inglês (garystafford/deepfake-audio-detection, CC BY 4.0)
#   Esse dataset também tem 933 REAIS em inglês (YouTube): entram nos reais em inglês (ver coletar_reais), senão a CNN
#   aprenderia o atalho "inglês = falso".
# Qualquer outra subpasta com áudios (ex.: playht/, murf/) também vira um "gerador comercial".
# Os áudios do PRÓPRIO grupo (./fake, ./original) NÃO entram aqui: são só o teste cego (seção 7).
PASTA_COMERCIAIS = f"{PASTA_PROJETO}/dados_comerciais"
# Falsos de ferramentas que devem ficar SÓ na generalização (ex.: Respeecher do Mendeley, baixado à mão)
PASTA_GENERALIZACAO = f"{PASTA_PROJETO}/dados_generalizacao"
EXTENSOES = (".wav", ".mp3", ".ogg", ".opus", ".m4a", ".flac", ".aac", ".webm", ".mp4", ".mpeg")
COMERCIAIS_HF = "garystafford/deepfake-audio-detection"
PREFIXOS_HF = {"el": "hf_elevenlabs", "po": "hf_polly", "sp": "hf_speechify", "hu": "hf_hume",
               "lv": "hf_luvvoice", "hg": "hf_kokoro"}
PASTA_REAIS_EN = f"{PASTA_PROJETO}/dados_reais_extra/youtube_en"

def _arquivos_locais(pasta):
    return sorted(os.path.join(r, n) for r, _, ns in os.walk(pasta) for n in ns if n.lower().endswith(EXTENSOES))

def geradores_locais(base):
    if not os.path.isdir(base):
        return []
    return sorted(g for g in os.listdir(base) if _arquivos_locais(os.path.join(base, g)))

def _ler_local(caminho):
    try:
        with open(caminho, "rb") as f:
            return f.read()
    except Exception:
        return None

def frases_common_voice(n, rng):
    """Frases do Common Voice (as mesmas que os voluntários reais leem): o texto não vira pista da classe."""
    shards = [f for f in list_repo_files(CV, repo_type="dataset")
              if f.startswith("pt/") and f.endswith(".parquet") and _e_do_split(f, "train")]
    frases = set()
    for sh in rng.permutation(shards):
        with abrir_parquet(CV, sh) as f:
            frases |= {s.strip() for s in pq.read_table(f, columns=["sentence"]).column("sentence").to_pylist()
                       if s and 20 <= len(s.strip()) <= 200}
        if len(frases) >= 2 * n:
            break
    return [str(s) for s in rng.permutation(sorted(frases))[:n]]

def _frases_novas(pasta, n, rng):
    """Frases ainda não geradas nesta pasta (o nome do arquivo leva o hash do texto): ao aumentar N, a mesma
    semente sortearia as mesmas frases e sobrescreveria arquivos, sem aumentar o total."""
    feitos = {os.path.splitext(os.path.basename(c))[0].rsplit("_", 1)[-1] for c in _arquivos_locais(pasta)}
    novas = [t for t in frases_common_voice(2 * n + len(feitos), rng)
             if hashlib.md5(t.encode()).hexdigest()[:12] not in feitos]
    return novas[:n]

def gerar_edge_tts(n, rng):
    import asyncio, edge_tts
    pasta = f"{PASTA_COMERCIAIS}/edge_tts"; os.makedirs(pasta, exist_ok=True)
    faltam = n - len(_arquivos_locais(pasta))
    if faltam <= 0:
        return
    frases = _frases_novas(pasta, faltam, rng)
    velocidades = [f"{int(v):+d}%" for v in rng.integers(-15, 16, size=len(frases))]

    async def tudo():
        vozes = sorted(v["ShortName"] for v in await edge_tts.list_voices() if v["Locale"] in ("pt-BR", "pt-PT"))
        print(f"  edge-tts: gerando {len(frases)} áudios com {len(vozes)} vozes: {vozes}")
        limite = asyncio.Semaphore(8)

        async def um(i, texto):
            voz = vozes[i % len(vozes)]
            nome = f"{pasta}/{voz}_{hashlib.md5(texto.encode()).hexdigest()[:12]}.mp3"
            async with limite:
                try:
                    await edge_tts.Communicate(texto, voz, rate=velocidades[i]).save(nome)
                except Exception as e:
                    print("  edge-tts falhou:", type(e).__name__, str(e)[:100])
                    if os.path.exists(nome):
                        os.remove(nome)
        await asyncio.gather(*(um(i, t) for i, t in enumerate(frases)))
    # numa thread separada: no Colab/Jupyter já existe um event loop rodando
    with cf.ThreadPoolExecutor(1) as ex:
        ex.submit(asyncio.run, tudo()).result()

def gerar_gtts(n, rng):
    from gtts import gTTS
    pasta = f"{PASTA_COMERCIAIS}/gtts"; os.makedirs(pasta, exist_ok=True)
    faltam = n - len(_arquivos_locais(pasta))
    if faltam <= 0:
        return
    print(f"  gTTS: gerando {faltam} áudios")
    for texto in _frases_novas(pasta, faltam, rng):
        tld = str(rng.choice(["com.br", "pt"]))
        nome = f"{pasta}/{tld}_{hashlib.md5(texto.encode()).hexdigest()[:12]}.mp3"
        try:
            gTTS(texto, lang="pt", tld=tld).save(nome)
        except Exception as e:                          # limite de requisições do Google: para e segue
            print("  gTTS interrompido:", type(e).__name__, str(e)[:100])
            if os.path.exists(nome):
                os.remove(nome)
            break

def baixar_comerciais_hf():
    """Falsos separados por serviço (pelo prefixo do nome: el_ = ElevenLabs, po_ = Polly...) + reais em inglês."""
    if _arquivos_locais(f"{PASTA_COMERCIAIS}/hf_elevenlabs") and _arquivos_locais(PASTA_REAIS_EN):
        return
    from huggingface_hub import snapshot_download
    tmp = snapshot_download(COMERCIAIS_HF, repo_type="dataset", allow_patterns=["fake/*", "real/*"],
                            local_dir=f"{PASTA_PROJETO}/_hf_comerciais")
    n = 0
    for c in _arquivos_locais(f"{tmp}/fake"):
        pasta = PREFIXOS_HF.get(os.path.basename(c).split("_")[0])
        if pasta:
            os.makedirs(f"{PASTA_COMERCIAIS}/{pasta}", exist_ok=True)
            shutil.copy(c, f"{PASTA_COMERCIAIS}/{pasta}/{os.path.basename(c)}"); n += 1
    os.makedirs(PASTA_REAIS_EN, exist_ok=True)
    for c in _arquivos_locais(f"{tmp}/real"):
        shutil.copy(c, f"{PASTA_REAIS_EN}/{os.path.basename(c)}")
    shutil.rmtree(tmp, ignore_errors=True)
    if not n:
        raise FileNotFoundError(f"nenhum falso reconhecido em {COMERCIAIS_HF}")

def preparar_comerciais():
    rng = np.random.default_rng(SEMENTE + 10)
    for nome, fn in [("edge-tts", lambda: gerar_edge_tts(N_GERAR_EDGE_TTS, rng)),
                     ("gTTS", lambda: gerar_gtts(N_GERAR_GTTS, rng)),
                     ("comerciais do Hugging Face", baixar_comerciais_hf)]:
        try:
            fn()
        except Exception as e:
            print(f"⚠️ {nome} indisponível ({type(e).__name__}: {str(e)[:150]}). Seguindo sem ele.")
    contagem = {g: len(_arquivos_locais(f"{PASTA_COMERCIAIS}/{g}")) for g in geradores_locais(PASTA_COMERCIAIS)}
    print("Áudios comerciais em disco:", contagem or "(nenhum)", "| reais em inglês:", len(_arquivos_locais(PASTA_REAIS_EN)))

IDIOMA_COMERCIAL = {pasta: "en" for pasta in PREFIXOS_HF.values()}      # edge_tts, gtts e outras: pt

def idioma_do_gerador(nome):
    """Idioma de um rótulo de gerador ("comercial:hf_polly", "mlaad:en/ElevenLabs-v3", "real_cv_en"...)."""
    if nome.startswith("mlaad:") and "/" in nome:
        return nome.split(":", 1)[1].split("/")[0]
    if nome.startswith("comercial:"):
        return IDIOMA_COMERCIAL.get(nome.split(":", 1)[1], "pt")
    if nome.startswith("real_cv_"):
        return nome.rsplit("_", 1)[1]
    if nome == "real:youtube_en":
        return "en"
    return "pt"

def plano_idiomas_falsos(split, n_por_classe):
    """Fração esperada de cada idioma (≠ pt) entre os falsos, calculada ANTES da coleta (os reais vêm primeiro)."""
    plano = {}
    if USAR_COMERCIAIS:
        disp = {g: len([p for p in _arquivos_locais(f"{PASTA_COMERCIAIS}/{g}") if _parte_mlaad(p) == split])
                for g in geradores_locais(PASTA_COMERCIAIS)}
        for g, q in planejar_vagas(int(round(n_por_classe * FRAC_COMERCIAL)), disp).items():
            idi = IDIOMA_COMERCIAL.get(g, "pt")
            plano[idi] = plano.get(idi, 0) + q
    if USAR_MLAAD:
        for idioma in MLAAD_GERADORES_EXTRA:
            plano[idioma] = plano.get(idioma, 0) + n_extra_idioma(n_por_classe)
    return {k: v / n_por_classe for k, v in plano.items() if k != "pt" and v}

def n_extra_idioma(n_por_classe):
    return int(round(n_por_classe * FRAC_MLAAD_OUTROS_IDIOMAS / max(1, len(MLAAD_GERADORES_EXTRA))))

def conferir_idiomas(grupos, split, tol=0.05):
    """Spec: a fração de cada idioma entre os reais fica a no máx. 5 pontos da fração entre os falsos."""
    fr = {}
    for rotulo, (_, gens) in grupos:
        s = pd.Series([idioma_do_gerador(str(g)) for g in gens])
        fr[rotulo] = s.value_counts(normalize=True) if len(s) else pd.Series(dtype=float)
    tab = pd.DataFrame({"reais": fr.get(0), "falsos": fr.get(1)}).fillna(0).round(3)
    print(f"  [{split}] idioma por classe:", tab.to_dict("index"))
    dif = (tab["reais"] - tab["falsos"]).abs().round(3)   # sem o round, 0,46 - 0,41 dá 0,0500000004 > 0,05
    assert (dif <= tol).all(), f"idioma desequilibrado entre as classes em {split}: {dif[dif > tol].to_dict()}"

def coletar_comerciais(split, n_total, rng, proc):
    gens = geradores_locais(PASTA_COMERCIAIS)
    if not gens:
        raise FileNotFoundError(f"nenhum áudio comercial em {PASTA_COMERCIAIS}")
    return coletar_por_gerador(split, n_total, rng, gens, lambda g: _arquivos_locais(f"{PASTA_COMERCIAIS}/{g}"),
                               _ler_local, "comercial", proc)

# ---------------- montagem dos conjuntos ----------------
def _com_reserva(fn_extra, fn_base, n_extra, n_total, nome):
    """Tenta a fonte extra; se falhar (ou vier menos), completa com a fonte base."""
    o, g = [], []
    if n_extra > 0:
        try:
            o, g = fn_extra(n_extra)
        except Exception as e:
            print(f"⚠️ {nome} indisponível ({type(e).__name__}: {str(e)[:150]}).")
        if len(o) < n_extra:
            print(f"⚠️ {nome}: vieram {len(o)} de {n_extra}; as {n_extra - len(o)} que faltaram vêm das outras fontes.")
    o2, g2 = fn_base(n_total - len(o))
    return o + o2, g + g2

def coletar_reais_en(split, n, rng, proc):
    if not _arquivos_locais(PASTA_REAIS_EN):
        raise FileNotFoundError(f"nenhum real em inglês em {PASTA_REAIS_EN}")
    return coletar_por_gerador(split, n, rng, ["youtube_en"], lambda g: _arquivos_locais(PASTA_REAIS_EN),
                               _ler_local, "real", proc)

def coletar_reais(split, n_total, rng, proc, idiomas=None):
    """Reais de 3 fontes: reais nos OUTROS idiomas dos falsos (`idiomas` = {idioma: fração}; inglês do YouTube
    primeiro, depois Common Voice do idioma) + CORAA (entrevistas) + Common Voice pt (celulares/notebooks),
    que completa o que faltar."""
    itens, geradores = [], []
    for idioma, frac in sorted((idiomas or {}).items()):
        n = int(round(n_total * frac))
        if idioma == "en" and USAR_COMERCIAIS and _arquivos_locais(PASTA_REAIS_EN):
            o, g = coletar_reais_en(split, n, rng, proc)
            itens += o; geradores += g; n -= len(o)
        if n > 0:
            try:
                o, g = coletar_common_voice(split, n, rng, proc, idioma)
                itens += o; geradores += g
            except Exception as e:
                print(f"⚠️ Common Voice {idioma} indisponível ({type(e).__name__}: {str(e)[:150]}).")
    for usar, frac, fn, nome in [(USAR_CORAA, FRAC_CORAA, coletar_coraa, "CORAA"),
                                 (USAR_COMMON_VOICE, 1.0, coletar_common_voice, "Common Voice")]:
        n = min(int(round(n_total * frac)), n_total - len(itens))
        if not usar or n <= 0:
            continue
        try:
            o, g = fn(split, n, rng, proc)
            itens += o; geradores += g
        except Exception as e:
            print(f"⚠️ {nome} indisponível ({type(e).__name__}: {str(e)[:150]}). Completando com as outras fontes.")
    if len(itens) < n_total:
        print(f"⚠️ Faltaram {n_total - len(itens)} reais (o conjunto fica um pouco desbalanceado).")
    return itens, geradores

def coletar_classes(split, n_por_classe, rng, proc):
    # REAIS primeiro: o piso de ruído deles é medido e depois emprestado aos falsos (anti-atalho 2)
    idiomas = plano_idiomas_falsos(split, n_por_classe)
    print(f"  [{split}] idiomas planejados (fração dos falsos): {idiomas}")
    reais = coletar_reais(split, n_por_classe, rng, lambda y: proc(y, 0), idiomas)
    pf = lambda y: proc(y, 1)
    gens = [g for g in MLAAD_GERADORES if g not in MLAAD_GERADORES_TESTE]
    # falsos em outros idiomas (MLAAD): cota fixa por idioma; o que faltar fica com o MLAAD pt
    extras_o, extras_g = [], []
    if USAR_MLAAD:
        for idioma, gens_idioma in MLAAD_GERADORES_EXTRA.items():
            o, g = coletar_mlaad(split, n_extra_idioma(n_por_classe), rng, gens_idioma, pf)
            extras_o += o; extras_g += g
    n_comercial = int(round(n_por_classe * FRAC_COMERCIAL)) if USAR_COMERCIAIS else 0
    # comerciais -> MLAAD: o que os comerciais não entregarem, o MLAAD completa
    def resto(n):
        return _com_reserva(lambda m: coletar_mlaad(split, m, rng, gens, pf), lambda m: ([], []),
                            n if USAR_MLAAD else 0, n, "MLAAD")
    falsos = _com_reserva(lambda n: coletar_comerciais(split, n, rng, pf), resto,
                          n_comercial, n_por_classe - len(extras_o), "Comerciais")
    grupos = [(0, reais), (1, (falsos[0] + extras_o, falsos[1] + extras_g))]
    conferir_idiomas(grupos, split)
    return grupos

def coletar_generalizacao(n_por_classe, rng, proc):
    """Teste de GENERALIZAÇÃO: falsos de geradores NUNCA vistos no treino x reais de teste (3 fontes).
    Falsos: os geradores reservados do MLAAD + as ferramentas em dados_generalizacao/ (se houver)."""
    reais = coletar_reais("test", n_por_classe, rng, lambda y: proc(y, 0))
    pf = lambda y: proc(y, 1)
    locais = geradores_locais(PASTA_GENERALIZACAO)
    gens = [g for g in MLAAD_GERADORES_TESTE if g in MLAAD_GERADORES] if USAR_MLAAD else []
    o, g = coletar_por_gerador("test", n_por_classe // 2 if gens else n_por_classe, rng, locais,
                               lambda x: _arquivos_locais(f"{PASTA_GENERALIZACAO}/{x}"), _ler_local, "gen", pf,
                               todas_as_partes=True)
    o2, g2 = coletar_mlaad("test", n_por_classe - len(o), rng, gens, pf, todas_as_partes=True)
    return [(0, reais), (1, (o + o2, g + g2))]

ITW_REPO, ITW_ZIP = "mueller91/In-The-Wild", "release_in_the_wild.zip"
PASTA_ITW = f"{PASTA_PROJETO}/dados_avaliacao/in_the_wild"

def baixar_in_the_wild(n_por_classe):
    """Copia n clipes de cada classe do ZIP (7,6 GB) para o disco, lendo SÓ esses arquivos (acesso aleatório)."""
    import zipfile
    if all(len(_arquivos_locais(f"{PASTA_ITW}/{c}")) >= n_por_classe for c in ("real", "fake")):
        return
    rng = np.random.default_rng(SEMENTE + 20)
    with FS.open(f"datasets/{ITW_REPO}/{ITW_ZIP}", "rb", block_size=2**20) as f:
        z = zipfile.ZipFile(f)
        meta = pd.read_csv(io.BytesIO(z.read("release_in_the_wild/meta.csv")))
        for rotulo, pasta in (("bona-fide", "real"), ("spoof", "fake")):
            os.makedirs(f"{PASTA_ITW}/{pasta}", exist_ok=True)
            arqs = rng.permutation(meta.loc[meta["label"] == rotulo, "file"].to_numpy())[:n_por_classe]
            for a in arqs:
                destino = f"{PASTA_ITW}/{pasta}/{a}"
                if not os.path.exists(destino):
                    with open(destino, "wb") as out:
                        out.write(z.read(f"release_in_the_wild/{a}"))
    print("In-the-Wild em disco:", {c: len(_arquivos_locais(f"{PASTA_ITW}/{c}")) for c in ("real", "fake")})

def coletar_in_the_wild(n_por_classe, rng, proc):
    baixar_in_the_wild(n_por_classe)
    lst = lambda c: _arquivos_locais(f"{PASTA_ITW}/{c}")
    reais = coletar_por_gerador("test", n_por_classe, rng, ["real"], lst, _ler_local, "itw", lambda y: proc(y, 0), True)
    falsos = coletar_por_gerador("test", n_por_classe, rng, ["fake"], lst, _ler_local, "itw", lambda y: proc(y, 1), True)
    return [(0, reais), (1, falsos)]

def construir_split(split, n_por_classe, modo):
    """modo='treino': 1 recorte aleatório, com codec (P_CODEC) e degradação (P_DEGRADAR) sorteados.
       modo='avaliacao': versão LIMPA e versão DEGRADADA (codec + degradação) do mesmo recorte.
    A extração das características roda em paralelo (threads: o ffmpeg e o numpy liberam o GIL)."""
    k_versoes = N_VERSOES_TREINO if modo == "treino" else 1
    # nome do cache = fontes e opções usadas ("50-" ficou da fração do MLAAD de versões anteriores: manter o
    # mesmo nome reaproveita os caches já calculados)
    tag = (f"{'M' if USAR_MLAAD else ''}{'K' if USAR_COMERCIAIS else ''}{'C' if USAR_CORAA else ''}"
           f"{'V' if USAR_COMMON_VOICE else ''}50-{int(100 * FRAC_COMERCIAL)}"
           f"{'R' if IGUALAR_PISO_DE_RUIDO else ''}"
           f"W{int(100 * P_WHATSAPP)}" + (f"K{k_versoes}" if modo == "treino" else "") +
           (f"L{int(100 * FRAC_MLAAD_OUTROS_IDIOMAS)}{'Q' if REPASSAR_VAGAS else ''}" if USAR_MLAAD else ""))
    cache = f"{PASTA_PROJETO}/cache_v8_mel_{split}_{n_por_classe}_{tag}.npz"
    if os.path.exists(cache):
        t0 = time.time()
        d = np.load(cache)
        dados = {k: d[k] for k in d.files}
        TEMPOS[f"{split}_leitura_cache_s"] = round(time.time() - t0, 1)
        print(f"[{split}] cache encontrado: {os.path.basename(cache)} ({TEMPOS[f'{split}_leitura_cache_s']:.0f}s)")
        return dados

    t0 = time.time()
    rng = np.random.default_rng(SEMENTE + {"train": 0, "validation": 1, "test": 2, "generalizacao": 3, "in_the_wild": 4}[split])
    pisos_reais = []
    pool = cf.ThreadPoolExecutor(N_TRABALHADORES)
    vagas = threading.BoundedSemaphore(4 * N_TRABALHADORES)       # limita trechos esperando na fila (RAM)

    def feat(t):
        return caracteristicas(t).astype(np.float16)

    def tarefa(versoes):
        """versoes = [(trecho, piso_alvo, semente), ...]. Treino: K versões independentes.
        Devolve (feats, feat_degradada ou None, canais)."""
        try:
            feats, canais = [], []
            for trecho, piso_alvo, semente in versoes:
                r = np.random.default_rng(semente)
                if piso_alvo is not None:
                    trecho = igualar_ruido(trecho, piso_alvo, r)
                if modo == "treino":
                    t, canal = canal_aleatorio(trecho, r)
                    feats.append(feat(t)); canais.append(canal)
                else:
                    # avaliação: falsos ficam como o gerador entregou (sem ruído emprestado) -> teste honesto
                    t, canal = canal_aleatorio(trecho, r, p_codec=1.0, p_degradar=1.0)
                    return ([feat(trecho)], feat(t), [canal])
            return (feats, None, canais)
        except Exception as e:
            print("  erro ao processar amostra:", e)
            return None
        finally:
            vagas.release()

    def proc(onda, rotulo):                           # onda inteira -> Future de (feats, feat_degradada, canais)
        versoes = []
        for _ in range(k_versoes):                    # cada versão: recorte, ruído emprestado e semente próprios
            trecho = recorte_aleatorio(onda, rng)
            piso_alvo = None
            if modo == "treino":
                if rotulo == 0:
                    pisos_reais.append(piso_ruido_db(trecho))
                elif IGUALAR_PISO_DE_RUIDO and pisos_reais and rng.random() < P_IGUALAR_PISO:
                    piso_alvo = float(rng.choice(pisos_reais))
            versoes.append((trecho, piso_alvo, int(rng.integers(2**31))))
        vagas.acquire()
        return pool.submit(tarefa, versoes)

    if split == "generalizacao":
        grupos = coletar_generalizacao(n_por_classe, rng, proc)
    elif split == "in_the_wild":
        grupos = coletar_in_the_wild(n_por_classe, rng, proc)
    else:
        grupos = coletar_classes(split, n_por_classe, rng, proc)
    t_coleta = time.time() - t0
    grupos = [(rotulo, [(f.result(), g) for f, g in zip(itens, geradores)]) for rotulo, (itens, geradores) in grupos]
    pool.shutdown()
    grupos = [(rotulo, [(r, g) for r, g in pares if r is not None]) for rotulo, pares in grupos]

    total = sum(len(pares) for _, pares in grupos)
    # pré-alocado (sem cópias extras). Treino: (N, K, linhas, quadros, 1); avaliação: (N, linhas, quadros, 1)
    forma = (total, k_versoes) if modo == "treino" else (total,)
    X = np.empty(forma + FORMA, dtype=np.float16)
    Xd = np.empty((total,) + FORMA, dtype=np.float16) if modo != "treino" else None
    y = np.empty(total, dtype=np.int64); gen, canais = [], []
    k = 0
    for rotulo, pares in grupos:
        for (ms, md, cs), g in pares:
            if modo == "treino":
                X[k] = np.stack(ms)
            else:
                X[k] = ms[0]; Xd[k] = md
            y[k] = rotulo; gen.append(g); canais.append(cs); k += 1
        pares.clear()
    del grupos; gc.collect()
    TEMPOS[f"{split}_coleta_s"] = round(t_coleta, 1)
    TEMPOS[f"{split}_caracteristicas_s"] = round(time.time() - t0 - t_coleta, 1)
    print(f"[{split}] coleta {t_coleta:.0f}s + características {TEMPOS[f'{split}_caracteristicas_s']:.0f}s")
    # (sem embaralhar aqui: o gerador de treino já embaralha a cada época)
    canais = np.array(canais)                                   # (N, K) no treino; (N, 1) na avaliação
    for rotulo, nome in ((0, "reais"), (1, "falsos")):
        c = canais[y == rotulo].ravel()
        if len(c):
            print(f"  [{split}] canal dos {nome}: " + ", ".join(f"{v or 'sem codec'} {np.mean(c == v):.1%}"
                                                          for v in ("", "codec", "whatsapp")))
    dados = {"X": X, "y": y, "gen": np.array(gen), "canal": canais}
    if Xd is not None:
        dados["Xd"] = Xd
    np.savez(cache, **dados)
    print(f"[{split}] {total} amostras, {X.nbytes / 2**20 * (2 if Xd is not None else 1):.0f} MB")
    return dados

MLAAD_GERADORES = []
MLAAD_GERADORES_EXTRA = {}
if USAR_MLAAD:
    try:
        MLAAD_GERADORES = mlaad_geradores()
        print(f"MLAAD (pt): {len(MLAAD_GERADORES)} geradores ->", MLAAD_GERADORES)
        USAR_MLAAD = testar_acesso_mlaad(MLAAD_GERADORES)
        if USAR_MLAAD:
            print("  ✅ acesso OK. Reservados só para o teste de generalização:", MLAAD_GERADORES_TESTE)
            MLAAD_GERADORES_EXTRA = mlaad_geradores_extra() if FRAC_MLAAD_OUTROS_IDIOMAS > 0 else {}
            print("  Outros idiomas:", {k: len(v) for k, v in MLAAD_GERADORES_EXTRA.items()}, "geradores")
    except Exception as e:
        print(f"⚠️ Não foi possível acessar o MLAAD ({str(e)[:150]}). Seguindo sem ele.")
        USAR_MLAAD = False

if USAR_COMERCIAIS:
    _t = time.time()
    preparar_comerciais()
    print(f"(preparação dos comerciais: {time.time() - _t:.0f}s)")

# %%
treino = construir_split("train",      N_TREINO_POR_CLASSE, "treino")
val    = construir_split("validation", N_VAL_POR_CLASSE,    "avaliacao")
teste  = construir_split("test",       N_TESTE_POR_CLASSE,  "avaliacao")

X_treino, y_treino = treino["X"], treino["y"]
# Validação mistura limpo + degradado: o early stopping passa a premiar robustez
X_val = np.concatenate([val["X"], val["Xd"]]); y_val = np.concatenate([val["y"], val["y"]])
X_teste, X_teste_deg, y_teste = teste["X"], teste["Xd"], teste["y"]

for nome, X_, y_ in [("treino", X_treino, y_treino), ("val", X_val, y_val), ("teste", X_teste, y_teste)]:
    print(f"{nome:7s} X={X_.shape}  [reais, falsos]={np.bincount(y_, minlength=2)}")
print("\nGeradores no treino:", pd.Series(treino["gen"]).value_counts().to_dict())
print("Geradores no teste: ", pd.Series(teste["gen"]).value_counts().to_dict())
gen_teste = teste["gen"]

# Conjunto de GENERALIZAÇÃO (geradores nunca vistos no treino)
generalizacao = None
if (USAR_MLAAD and MLAAD_GERADORES_TESTE) or geradores_locais(PASTA_GENERALIZACAO):
    generalizacao = construir_split("generalizacao", N_GENERALIZACAO_POR_CLASSE, "avaliacao")
    print("Generalização:", generalizacao["X"].shape, np.bincount(generalizacao["y"], minlength=2),
          pd.Series(generalizacao["gen"]).value_counts().to_dict())

# Conjunto IN-THE-WILD (internet, inglês): nunca entra no treino nem no limiar
in_the_wild = None
if USAR_IN_THE_WILD:
    try:
        in_the_wild = construir_split("in_the_wild", N_IN_THE_WILD_POR_CLASSE, "avaliacao")
        print("In-the-Wild:", in_the_wild["X"].shape, np.bincount(in_the_wild["y"], minlength=2))
    except Exception as e:
        print(f"⚠️ In-the-Wild indisponível ({type(e).__name__}: {str(e)[:150]}). Seguindo sem essa avaliação.")

# %%
# Exemplo visual: real x falso, limpo x degradado
i_r, i_f = int(np.where(y_teste == 0)[0][0]), int(np.where(y_teste == 1)[0][0])
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
for col, (i, nome) in enumerate([(i_r, "REAL"), (i_f, "FALSO")]):
    for lin, (Xs, cond) in enumerate([(X_teste, "limpo"), (X_teste_deg, "degradado")]):
        axes[lin, col].imshow(Xs[i, :, :, 0].astype(np.float32), aspect="auto", origin="lower", cmap="magma")
        axes[lin, col].set_title(f"{nome} — {cond}")
        axes[lin, col].set_xlabel("tempo (quadros de 10 ms)"); axes[lin, col].set_ylabel("banda mel")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 3. Arquitetura da CNN
# 4 blocos residuais [Conv 3×3 → BN → ReLU → Conv 3×3 → BN] + atalho (Conv 1×1 + BN) → ReLU → MaxPool →
# SpatialDropout (filtros 32-64-96-128); **GlobalAveragePooling + GlobalMaxPooling** (média e pico) → Dense(64)
# → Dropout → Dense(1, **sigmoid**) = P(fake). O ensemble tira a média de 2 dessas redes (sementes diferentes).
# O pooling de máximo ajuda a pegar artefatos de síntese *localizados* no espectrograma, que a média dilui.
# Função de perda: entropia cruzada binária (com label smoothing). Otimizador: AdamW + agenda cosseno.

# %%
def ativacao_camada(nome):
    if nome == "leaky_relu":
        return layers.LeakyReLU(negative_slope=0.1)
    if nome == "relu":
        # LeakyReLU com inclinação 0 = ReLU (mesma conta). Contorna um defeito do tensorflow-metal 1.2.0: no grafo
        # (fit/predict), a sequência MatMul + BiasAdd + ReLU dá resultado ERRADO na GPU do Mac (erro de até 4 nos
        # valores; P(fake) até 0,1 diferente do cálculo correto). Na convolução o defeito não aparece.
        return layers.LeakyReLU(negative_slope=0.0)
    return layers.Activation(nome)

def _bloco_residual(x, filtros, ativacao):
    atalho = x
    if x.shape[-1] != filtros:
        atalho = layers.BatchNormalization(momentum=0.9)(layers.Conv2D(filtros, 1, use_bias=False)(x))
    y = layers.Conv2D(filtros, 3, padding="same", use_bias=False)(x)
    y = ativacao_camada(ativacao)(layers.BatchNormalization(momentum=0.9)(y))
    y = layers.BatchNormalization(momentum=0.9)(layers.Conv2D(filtros, 3, padding="same", use_bias=False)(y))
    return ativacao_camada(ativacao)(layers.Add()([y, atalho]))

def construir_cnn(ativacao="relu", nome="cnn_deepfake"):
    entrada = layers.Input(shape=FORMA)
    x = entrada
    for filtros in (32, 64, 96, 128):
        x = _bloco_residual(x, filtros, ativacao)
        x = layers.SpatialDropout2D(0.1)(layers.MaxPooling2D(2)(x))
    x = layers.Concatenate()([layers.GlobalAveragePooling2D()(x), layers.GlobalMaxPooling2D()(x)])
    x = layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = ativacao_camada(ativacao)(x)
    x = layers.Dropout(0.4)(x)
    saida = layers.Dense(1, activation="sigmoid", dtype="float32")(x)   # float32 mesmo com precisão mista
    return models.Model(entrada, saida, name=nome)

def montar_ensemble(modelos_):
    """Um único Model que devolve a média das P(fake): o resto do notebook o usa como se fosse um modelo só."""
    if len(modelos_) == 1:
        return modelos_[0]
    assert len({m.name for m in modelos_}) == len(modelos_), "dê nomes diferentes aos modelos do ensemble"
    entrada = layers.Input(shape=modelos_[0].input_shape[1:])
    return models.Model(entrada, layers.Average()([m(entrada) for m in modelos_]), name="ensemble")

construir_cnn().summary()

# %%
TEMPO_MASK_MAX = 30                     # até 0,3 s de 3 s (a máscara de frequência acompanha a altura da entrada)

@tf.keras.utils.register_keras_serializable(package="deepfake")
class SpecAugment(layers.Layer):
    """SpecAugment em lote, DENTRO do grafo (roda na GPU, compatível com XLA, só ativo no treino).
    Zera uma faixa aleatória de bandas mel e uma de quadros de tempo, diferente para cada amostra."""
    def __init__(self, freq_max=15, tempo_max=40, **kw):
        super().__init__(**kw)
        self.freq_max, self.tempo_max = freq_max, tempo_max

    @staticmethod
    def _mascara(n, largura_max, lote):
        w = tf.random.uniform([lote, 1], 0, largura_max + 1, dtype=tf.int32)
        ini = tf.cast(tf.random.uniform([lote, 1]) * tf.cast(n - w + 1, tf.float32), tf.int32)
        idx = tf.range(n)[None, :]
        return tf.cast(tf.logical_not((idx >= ini) & (idx < ini + w)), tf.float32)      # (lote, n)

    def call(self, x, training=None):
        if not training:
            return x
        lote = tf.shape(x)[0]
        mf = self._mascara(x.shape[1], self.freq_max, lote)[:, :, None, None]
        mt = self._mascara(x.shape[2], self.tempo_max, lote)[:, None, :, None]
        return x * tf.cast(mf * mt, x.dtype)

    def get_config(self):
        return {**super().get_config(), "freq_max": self.freq_max, "tempo_max": self.tempo_max}

def criar_ds(X, y, embaralhar=True, batch_size=None, gerador_python=None):
    """Lotes de treino/avaliação. Se os dados cabem na RAM (LIMITE_DS_NA_RAM_GB), o tf.data sorteia os índices
    e junta o lote com tf.gather em threads próprias, sem laço Python no caminho da GPU. Senão, gerador Python.
    Treino: embaralha a cada época e descarta o lote incompleto (lotes de tamanho fixo, bom p/ XLA).
    X com 5 dimensões (N, K, ...): a cada lote, sorteia UMA das K versões de cada amostra (muda a cada época)."""
    batch_size = batch_size or BATCH_SIZE
    if gerador_python is None:
        gerador_python = X.nbytes > LIMITE_DS_NA_RAM_GB * 2**30
    if gerador_python:
        ds = _criar_ds_gerador(X, y, embaralhar, batch_size)
    else:
        with tf.device("/CPU:0"):
            Xt, yt = tf.constant(X), tf.constant(y.astype("float32"))
        k = X.shape[1] if X.ndim == 5 else 0

        def lote(i):
            if k:
                v = tf.random.uniform(tf.shape(i), 0, k, dtype=tf.int64)
                return tf.gather_nd(Xt, tf.stack([i, v], axis=1)), tf.gather(yt, i)
            return tf.gather(Xt, i), tf.gather(yt, i)
        ds = tf.data.Dataset.range(len(X))
        if embaralhar:
            ds = ds.shuffle(len(X), reshuffle_each_iteration=True)
        ds = ds.batch(batch_size, drop_remainder=embaralhar).map(lote, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

def _criar_ds_gerador(X, y, embaralhar, batch_size):
    """Versão anterior (gerador Python): fica para dados maiores que a RAM permite copiar para o tf.data."""
    n, y = len(X), y.astype("float32")
    forma_lote = X.shape[2:] if X.ndim == 5 else X.shape[1:]
    def gerador():
        idx = np.random.permutation(n) if embaralhar else np.arange(n)
        fim = n - n % batch_size if embaralhar else n            # treino: lotes de tamanho fixo (bom p/ XLA)
        for i in range(0, fim, batch_size):
            b = np.sort(idx[i:i + batch_size]) if embaralhar else idx[i:i + batch_size]
            if X.ndim == 5:                                       # uma versão sorteada por amostra
                yield X[b, np.random.randint(X.shape[1], size=len(b))], y[b]
            else:
                yield X[b], y[b]
    ds = tf.data.Dataset.from_generator(gerador, output_signature=(
        tf.TensorSpec(shape=(None,) + forma_lote, dtype=tf.as_dtype(X.dtype)),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)))
    n_lotes = n // batch_size if embaralhar else int(np.ceil(n / batch_size))
    return ds.apply(tf.data.experimental.assert_cardinality(n_lotes))

def otimizador(nome, lr_pico, passos_totais, passos_aquec):
    """Aquecimento linear (1 época) + decaimento cosseno até 2% do pico, no lugar do ReduceLROnPlateau."""
    agenda = tf.keras.optimizers.schedules.CosineDecay(
        0.0, decay_steps=max(1, passos_totais - passos_aquec), alpha=0.02,
        warmup_target=lr_pico, warmup_steps=passos_aquec)
    if nome == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=agenda, momentum=0.9, nesterov=True)
    if nome == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=agenda)
    return tf.keras.optimizers.AdamW(learning_rate=agenda, weight_decay=1e-4)

class SalvarMelhor(tf.keras.callbacks.Callback):
    """Salva o modelo SEM a camada de aumento (é ele que o classificador carrega depois)."""
    def __init__(self, modelo, caminho):
        super().__init__(); self.modelo, self.caminho, self.melhor = modelo, caminho, np.inf
    def on_epoch_end(self, epoch, logs=None):
        if logs and logs.get("val_loss", np.inf) < self.melhor:
            self.melhor = logs["val_loss"]; self.modelo.save(self.caminho)

class TempoEpoca(tf.keras.callbacks.Callback):
    """Guarda a duração de cada época (em hist.history["tempo_s"])."""
    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs["tempo_s"] = time.time() - self.t0

def calcular_eer(y_true, scores):
    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(np.clip(thr[i], 0, 1))

def treinar(modelo, nome, X_tr, y_tr, epocas=EPOCAS_MAX, lr=None, opt="adam", paciencia=PACIENCIA, salvar=True,
            validacao=None):
    lr = LR_PICO if lr is None else lr
    # O SpecAugment fica num "invólucro" que compartilha os pesos de `modelo`: treinamos o invólucro,
    # mas o modelo salvo/avaliado é o original, sem a camada de aumento.
    entrada = layers.Input(shape=modelo.input_shape[1:])
    freq_max = max(1, modelo.input_shape[1] // 10)          # mel 128 -> até 12 bandas
    treino_m = models.Model(entrada, modelo(SpecAugment(freq_max, TEMPO_MASK_MAX)(entrada)))
    passos = len(X_tr) // BATCH_SIZE
    xla = USAR_XLA and CUDA
    cbs = [TempoEpoca(),
           tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=paciencia, restore_best_weights=True)]
    if salvar:
        cbs.append(SalvarMelhor(modelo, f"{PASTA_PROJETO}/melhor_{nome}.keras"))
    Xv, yv = validacao if validacao is not None else (X_val, y_val)
    ds_val = criar_ds(Xv, yv, embaralhar=False, batch_size=128)

    def compilar_e_treinar(com_xla):
        treino_m.compile(optimizer=otimizador(opt, lr, epocas * passos, passos),
                         loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
                         metrics=["accuracy", tf.keras.metrics.AUC(name="auc")], jit_compile=com_xla)
        return treino_m.fit(criar_ds(X_tr, y_tr), validation_data=ds_val,
                            epochs=epocas, callbacks=cbs, verbose=2)
    try:
        hist = compilar_e_treinar(xla)
    except Exception as e:
        if not xla:
            raise
        print(f"⚠️ XLA falhou ({type(e).__name__}); repetindo sem XLA.")
        hist = compilar_e_treinar(False)
    n_ep = len(hist.history["loss"])
    melhor = int(np.argmin(hist.history["val_loss"])) + 1
    if n_ep < epocas:
        print(f"[{nome}] parada antecipada na época {n_ep} (melhor: {melhor}).")
    elif paciencia <= epocas:              # (paciência > épocas = rodada de duração fixa, sem aviso)
        print(f"⚠️ [{nome}] chegou ao máximo de {epocas} épocas sem parada antecipada (melhor: {melhor}): "
              "o modelo pode não ter convergido.")
    return hist

def plotar_curvas(hist, titulo):
    h = hist.history
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for a, m, t in zip(ax, ["loss", "auc", "accuracy"], ["Loss (BCE)", "AUC", "Acurácia"]):
        a.plot(h[m], label="treino"); a.plot(h["val_" + m], label="validação")
        a.set_title(f"{titulo} — {t}"); a.set_xlabel("época"); a.legend()
    plt.tight_layout(); plt.show()

# %% [markdown]
# ## 4. Experimento: funções de ativação e otimizadores
# Mesmo dado, mesma arquitetura, só muda a ativação (ReLU, LeakyReLU, ELU) ou o otimizador
# (Adam, SGD+momentum, RMSprop). Tabela vai direto para o Resumo Expandido.

# %%
resultados_comparacao = []
melhor_config = {"ativacao": "relu", "opt": "adam"}
if COMPARAR_CONFIGURACOES:
    configs = [("relu", "adam"), ("leaky_relu", "adam"), ("elu", "adam"),
               ("relu", "sgd"), ("relu", "rmsprop")]
    for ativ, opt in configs:
        reiniciar_sessao()
        m = construir_cnn(ativacao=ativ)
        t0 = time.time()
        h = treinar(m, f"cmp_{ativ}_{opt}", X_treino, y_treino, epocas=EPOCAS_COMPARACAO,
                    opt=opt, paciencia=4, salvar=False)
        p_val = m.predict(X_val, batch_size=64, verbose=0).ravel()
        p_deg = m.predict(X_teste_deg, batch_size=64, verbose=0).ravel()
        resultados_comparacao.append({
            "ativacao": ativ, "otimizador": opt, "epocas": len(h.history["loss"]),
            "val_loss_min": float(np.min(h.history["val_loss"])),
            "auc_val": float(roc_auc_score(y_val, p_val)),
            "auc_teste_degradado": float(roc_auc_score(y_teste, p_deg)),
            "tempo_s": round(time.time() - t0)})
        del m; gc.collect()
    df_cmp = pd.DataFrame(resultados_comparacao).sort_values("val_loss_min")
    print(df_cmp.round(4).to_string(index=False))
    df_cmp.to_csv(f"{PASTA_PROJETO}/comparacao_ativacoes_otimizadores.csv", index=False)
    melhor = df_cmp.iloc[0]
    melhor_config = {"ativacao": melhor["ativacao"], "opt": melhor["otimizador"]}
print("Configuração escolhida (menor val_loss):", melhor_config)


# %% [markdown]
# ## 5. Treino do modelo final

# %%
print("Política de precisão no treino:", tf.keras.mixed_precision.global_policy().name, "| lote:", BATCH_SIZE,
      "| modelos no ensemble:", N_MODELOS_ENSEMBLE)
# Ensemble: mesmos dados e configuração, só muda a semente (inicialização e ordem dos lotes).
# Com 1 modelo, os arquivos são os de sempre (modelo_cnn_final.keras / melhor_cnn_final.keras).
modelos_ensemble, MODELOS_SALVOS = [], []
for _s in range(N_MODELOS_ENSEMBLE):
    _sufixo = f"_s{_s}" if N_MODELOS_ENSEMBLE > 1 else ""
    reiniciar_sessao(SEMENTE + _s)                      # o 1º modelo usa a semente de sempre
    _m = construir_cnn(ativacao=melhor_config["ativacao"], nome=f"cnn_deepfake{_sufixo}")
    hist_cnn = treinar(_m, f"cnn_final{_sufixo}", X_treino, y_treino, opt=melhor_config["opt"])
    _t_ep = hist_cnn.history["tempo_s"]
    TEMPOS.update({f"treino{_sufixo}_epocas": len(_t_ep), f"treino{_sufixo}_total_s": round(sum(_t_ep), 1),
                   f"treino{_sufixo}_s_por_epoca": [round(t, 1) for t in _t_ep],
                   # 1ª época fora da média: inclui a compilação do grafo
                   f"treino{_sufixo}_amostras_por_s":
                       round((len(y_treino) // BATCH_SIZE) * BATCH_SIZE / np.median(_t_ep[1:] or _t_ep))})
    print(f"Treino{_sufixo}: {len(_t_ep)} épocas em {sum(_t_ep) / 60:.1f} min")
    _m.save(f"{PASTA_PROJETO}/modelo_cnn_final{_sufixo}.keras")
    MODELOS_SALVOS.append(f"modelo_cnn_final{_sufixo}.keras")
    modelos_ensemble.append(_m)
    plotar_curvas(hist_cnn, f"CNN final{_sufixo}")
modelo_cnn = montar_ensemble(modelos_ensemble)

# %% [markdown]
# ## 6. Avaliação no teste (limpo e degradado)
# Métricas: acurácia, precisão/recall/F1, AUC e **EER** (taxa de erro igual — métrica padrão
# em anti-spoofing; quanto menor, melhor).

# %%
resultados = {}

def avaliar(modelo, X, y, titulo, mostrar=True):
    t0 = time.time()
    p = modelo.predict(X, batch_size=64, verbose=0).ravel()
    TEMPOS[f"predicao_{titulo}_s"] = round(time.time() - t0, 2)
    pred = (p >= LIMIAR).astype(int)
    eer, lim_eer = calcular_eer(y, p)
    rel = classification_report(y, pred, labels=[0, 1], target_names=["real", "fake"],
                                zero_division=0, output_dict=True)
    r = {"acuracia": float((pred == y).mean()), "precisao_fake": rel["fake"]["precision"],
         "recall_fake": rel["fake"]["recall"], "recall_real": rel["real"]["recall"],
         "f1_macro": rel["macro avg"]["f1-score"], "auc": float(roc_auc_score(y, p)),
         "eer": eer, "limiar_eer": lim_eer}
    resultados[titulo] = r
    if mostrar:
        print("=" * 64, f"\n{titulo}  (limiar {LIMIAR})\n" + "=" * 64)
        print(classification_report(y, pred, labels=[0, 1], target_names=["real", "fake"], zero_division=0))
        print(f"AUC = {r['auc']:.4f} | EER = {100*eer:.2f}% (limiar do EER = {lim_eer:.3f})")
        fig, ax = plt.subplots(1, 3, figsize=(17, 4))
        sns.heatmap(confusion_matrix(y, pred, labels=[0, 1]), annot=True, fmt="d", cmap="Blues", cbar=False,
                    ax=ax[0], xticklabels=["real", "fake"], yticklabels=["real", "fake"])
        ax[0].set_title(f"Matriz de confusão — {titulo}"); ax[0].set_xlabel("Previsto"); ax[0].set_ylabel("Verdadeiro")
        fpr, tpr, _ = roc_curve(y, p)
        ax[1].plot(fpr, tpr, label=f"AUC = {r['auc']:.3f}"); ax[1].plot([0, 1], [0, 1], "--", color="gray")
        ax[1].set_title("Curva ROC"); ax[1].set_xlabel("Taxa de falso positivo"); ax[1].set_ylabel("Taxa de verdadeiro positivo"); ax[1].legend()
        ax[2].hist(p[y == 0], bins=30, alpha=.6, label="real"); ax[2].hist(p[y == 1], bins=30, alpha=.6, label="fake")
        ax[2].axvline(LIMIAR, color="k", ls="--"); ax[2].set_title("Distribuição de P(fake)"); ax[2].legend()
        plt.tight_layout(); plt.show()
    return p

p_teste = avaliar(modelo_cnn, X_teste, y_teste, "Teste limpo")
p_teste_deg = avaliar(modelo_cnn, X_teste_deg, y_teste, "Teste degradado (canal)")

# Acerto por gerador (quais TTS são mais difíceis de detectar?)
df_gen = pd.DataFrame({"gerador": gen_teste, "limpo": (p_teste >= LIMIAR) == y_teste,
                       "degradado": (p_teste_deg >= LIMIAR) == y_teste})
tab_gen = df_gen.groupby("gerador")[["limpo", "degradado"]].mean().round(3)
tab_gen["n"] = df_gen.groupby("gerador").size()
print("\nTaxa de acerto por gerador (bonafide = voz real):"); print(tab_gen.to_string())
tab_gen.to_csv(f"{PASTA_PROJETO}/acerto_por_gerador.csv")

# Generalização: TTS que o modelo NUNCA viu (é o número que mais se aproxima do "mundo real")
if generalizacao is not None:
    p_gen = avaliar(modelo_cnn, generalizacao["X"], generalizacao["y"], "Generalização (TTS nunca vistos)")
    avaliar(modelo_cnn, generalizacao["Xd"], generalizacao["y"], "Generalização degradada", mostrar=False)
    df_g = pd.DataFrame({"gerador": generalizacao["gen"], "acerto": (p_gen >= LIMIAR) == generalizacao["y"],
                         "P(fake) média": p_gen})
    print(df_g.groupby("gerador")[["acerto", "P(fake) média"]].mean().round(3).to_string())

# Ensemble: EER de cada modelo sozinho x EER da média (spec: mostrar os dois)
if len(modelos_ensemble) > 1:
    _conj = [("Teste limpo", X_teste, y_teste), ("Teste degradado", X_teste_deg, y_teste)]
    if generalizacao is not None:
        _conj.append(("Generalização", generalizacao["X"], generalizacao["y"]))
    _linhas = []
    for nome_m, m_ in [(f"modelo {i}", m_) for i, m_ in enumerate(modelos_ensemble)] + [("ENSEMBLE", modelo_cnn)]:
        _linhas.append({"modelo": nome_m, **{c: round(100 * calcular_eer(yy, m_.predict(XX, batch_size=128, verbose=0).ravel())[0], 2)
                                            for c, XX, yy in _conj}})
    df_ens = pd.DataFrame(_linhas); print("EER (%) por modelo e do ensemble:\n" + df_ens.to_string(index=False))
    resultados["ensemble_eer_por_modelo"] = df_ens.set_index("modelo").to_dict("index")

# In-the-Wild: áudio de internet (inglês), avaliação extra sem nunca ter entrado no treino nem no limiar
if in_the_wild is not None:
    avaliar(modelo_cnn, in_the_wild["X"], in_the_wild["y"], "In-the-Wild (internet)")

# ---------- Limiar de OPERAÇÃO: ponto de EER da VALIDAÇÃO (limpa + degradada), aplicado ao teste ----------
# Onde "real marcado como falso" = "falso aceito como real" (equilibrado, sem piso de 0,5).
LIMIAR_OPERACAO = calcular_eer(y_val, modelo_cnn.predict(X_val, batch_size=128, verbose=0).ravel())[1]
print(f"\nLimiar de OPERAÇÃO (EER da validação): {LIMIAR_OPERACAO:.3f} | margem INCONCLUSIVO: ±{MARGEM_INCERTA}")
conjuntos = [("Teste limpo", p_teste, y_teste), ("Teste degradado", p_teste_deg, y_teste)]
if generalizacao is not None:
    conjuntos.append(("Generalização", p_gen, generalizacao["y"]))

def taxas(p, yy, lim):
    return {"reais marcados FALSO (FPR)": float((p[yy == 0] >= lim).mean()),
            "falsos detectados (recall)": float((p[yy == 1] >= lim).mean())}

linhas = []
for n, p, yy in conjuntos:
    for nome_lim, lim in [("0.50", 0.5), (f"operação {LIMIAR_OPERACAO:.2f}", LIMIAR_OPERACAO)]:
        linhas.append({"conjunto": n, "limiar": nome_lim, **taxas(p, yy, lim)})
tab_op = pd.DataFrame(linhas).round(3)
print(tab_op.to_string(index=False))

# Falso alarme POR FONTE de voz real: se uma fonte (ex.: celulares do Common Voice) tiver FPR alto,
# é sinal de que o modelo ainda confunde "tipo de gravação" com "voz sintética".
fontes_reais = [g for g in np.unique(gen_teste) if not np.any(y_teste[gen_teste == g] == 1)]
tab_fpr = pd.DataFrame([{"fonte real": g, "n": int((gen_teste == g).sum()),
                         "FPR limpo @op": float((p_teste[gen_teste == g] >= LIMIAR_OPERACAO).mean()),
                         "FPR degradado @op": float((p_teste_deg[gen_teste == g] >= LIMIAR_OPERACAO).mean()),
                         "P(fake) média": float(p_teste[gen_teste == g].mean())} for g in fontes_reais]).round(3)
print("\nVozes REAIS marcadas como falsas, por fonte:\n" + tab_fpr.to_string(index=False))
tab_fpr.to_csv(f"{PASTA_PROJETO}/falso_alarme_por_fonte_real.csv", index=False)
tab_op.to_csv(f"{PASTA_PROJETO}/taxas_no_limiar_de_operacao.csv", index=False)
resultados["limiar_operacao"] = {"limiar": LIMIAR_OPERACAO, "criterio": "eer_validacao"}

# %% [markdown]
# ## 7. Teste CEGO com os áudios do grupo
# `./fake/` (vozes de IA: ElevenLabs e outros sites) e `./original/` (gravações reais, ex.: WhatsApp).
# No Colab, use `teste_cego/fake/` e `teste_cego/original/` dentro da pasta do projeto no Drive.
# Esses arquivos NUNCA entram no treino nem na calibração do limiar: é a prova mais honesta do "mundo real".
# Cada áudio é dividido em janelas de 3 s (passo 1,5 s); P(fake) do arquivo = média das janelas, comparada ao
# **limiar de operação** (seção 6). Perto do limiar, o veredito é "INCONCLUSIVO".

# %%
def carregar_arquivo(caminho):
    """Decodifica qualquer formato (mp3, wav, ogg/opus do WhatsApp, m4a...) via ffmpeg."""
    bruto = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", caminho,
                            "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
                           stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(bruto, dtype=np.float32).copy()

def mels_do_arquivo(caminho):
    """Arquivo -> características de todas as janelas (mesmo pré-processamento do treino)."""
    y = preparar_onda(carregar_arquivo(caminho), SR)
    if len(y) < DUR_MIN * SR:
        raise ValueError(f"menos de {DUR_MIN:.0f} s de fala depois de remover o silêncio")
    return np.stack([caracteristicas(j) for j in janelas(y)]), len(y) / SR

def veredito(p, limiar=None):
    limiar = LIMIAR_OPERACAO if limiar is None else limiar
    if abs(p - limiar) < MARGEM_INCERTA:
        return "INCONCLUSIVO"
    return "FALSO (sintético)" if p >= limiar else "REAL"

PASTA_TESTE_CEGO = "." if (os.path.isdir("fake") or os.path.isdir("original")) else f"{PASTA_PROJETO}/teste_cego"

def avaliar_arquivos(modelo, itens):
    """P(fake) por ARQUIVO (média das janelas de 3 s)."""
    linhas = []
    for c, r, f in itens:
        try:
            probs = modelo.predict(mels_do_arquivo(c)[0], verbose=0).ravel()
        except Exception as e:
            print("Ignorado:", c, e); continue
        linhas.append({"arquivo": os.path.basename(c), "fonte": f, "verdadeiro": "fake" if r else "real",
                       "p_fake": round(float(probs.mean()), 3), "janelas": len(probs)})
    df = pd.DataFrame(linhas)
    df["acertou"] = (df["p_fake"] >= LIMIAR_OPERACAO) == (df["verdadeiro"] == "fake")
    return df

def teste_cego(modelo, base=PASTA_TESTE_CEGO):
    itens = [(c, r, sub) for r, sub in [(0, "original"), (1, "fake")] for c in _arquivos_locais(f"{base}/{sub}")]
    if not itens:
        print(f"Teste cego pulado: nenhum áudio em {base}/fake nem em {base}/original.")
        return None
    df = avaliar_arquivos(modelo, itens)
    df["veredito"] = df["p_fake"].map(veredito)
    df["acertou"] = df["acertou"] & (df["veredito"] != "INCONCLUSIVO")
    print(df[["arquivo", "verdadeiro", "p_fake", "veredito", "acertou", "janelas"]].to_string(index=False))
    f, r = df[df.verdadeiro == "fake"], df[df.verdadeiro == "real"]
    resumo = {"falsos_detectados": f"{int(f.acertou.sum())}/{len(f)}", "reais_corretos": f"{int(r.acertou.sum())}/{len(r)}",
              "inconclusivos": int((df.veredito == "INCONCLUSIVO").sum()), "limiar": round(LIMIAR_OPERACAO, 3)}
    print("Resumo do teste cego:", resumo)
    df.to_csv(f"{PASTA_PROJETO}/teste_cego.csv", index=False)
    return resumo

def teste_formato(modelo, base=PASTA_TESTE_CEGO):
    """Mesmo áudio REAL salvo como WAV, MP3 e Ogg/Opus: P(fake) não deveria mudar muito só pelo formato."""
    reais = _arquivos_locais(f"{base}/original")
    if not reais:
        print("Teste de formato pulado: nenhum áudio em original/.")
        return None
    y = carregar_arquivo(reais[0])
    pasta = tempfile.mkdtemp()
    linhas = []
    for ext, codec in [("wav", ["-c:a", "pcm_s16le"]), ("mp3", ["-c:a", "libmp3lame", "-b:a", "128k"]),
                       ("ogg", ["-c:a", "libopus", "-b:a", "32k"])]:
        destino = f"{pasta}/mesmo_audio.{ext}"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1",
                        "-i", "pipe:0", *codec, destino], input=y.tobytes(), check=True)
        linhas.append({"formato": ext, "p_fake": float(modelo.predict(mels_do_arquivo(destino)[0], verbose=0).mean())})
    shutil.rmtree(pasta, ignore_errors=True)
    df = pd.DataFrame(linhas).round(3)
    dif = float(df.p_fake.max() - df.p_fake.min())
    print(f"Teste de formato ({os.path.basename(reais[0])}):\n{df.to_string(index=False)}\n"
          f"maior diferença de P(fake) entre formatos: {dif:.3f}")
    return {"arquivo": os.path.basename(reais[0]), "diferenca_max_p_fake": round(dif, 3),
            **{f"p_fake_{l['formato']}": l["p_fake"] for l in linhas}}

import tempfile
r_cego = teste_cego(modelo_cnn)
if r_cego:
    resultados["Teste cego (áudios do grupo)"] = r_cego
r_formato = teste_formato(modelo_cnn)
if r_formato:
    resultados["Teste de formato"] = r_formato

# %% [markdown]
# ## 8. Resultados consolidados e salvamento

# %%
tabela = pd.DataFrame(resultados).T
print(tabela.round(4).to_string())
with open(f"{PASTA_PROJETO}/resultados.json", "w", encoding="utf-8") as f:
    # config: o pré-processamento, o limiar e os modelos que o classificador (seção 9) lê
    json.dump({"config": {"sr": SR, "duracao_s": DURACAO, "passo_janela_s": PASSO_JANELA / SR, "hop": HOP,
                          "n_mels": N_MELS, "n_fft": N_FFT, "n_treino": int(len(y_treino)),
                          "ativacao": melhor_config["ativacao"], "otimizador": melhor_config["opt"],
                          "p_codec": P_CODEC, "p_degradar": P_DEGRADAR, "p_whatsapp": P_WHATSAPP,
                          "n_versoes_treino": N_VERSOES_TREINO, "limiar_operacao": LIMIAR_OPERACAO,
                          "criterio_limiar": "eer_validacao", "margem_incerta": MARGEM_INCERTA,
                          "n_modelos_ensemble": N_MODELOS_ENSEMBLE, "modelos": MODELOS_SALVOS,
                          "precisao": tf.keras.mixed_precision.global_policy().name, "batch_size": BATCH_SIZE,
                          "lr_pico": LR_PICO},
               "comparacao": resultados_comparacao, "resultados": resultados, "tempos": TEMPOS},
              f, indent=2, ensure_ascii=False, default=float)
print("Arquivos salvos em:", PASTA_PROJETO)

# %% [markdown]
# ## 9. Classificador: REAL ou FAKE
# Célula independente: carrega o modelo salvo no Drive (não precisa treinar de novo) e classifica
# os áudios enviados (mp3, wav, ogg/opus do WhatsApp, m4a...). Pode rodar numa sessão nova.

# %%
import os, json, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import tensorflow as tf

# ---------- configuração ----------
try:
    from google.colab import drive, files
    drive.mount("/content/drive")
    PASTA_PROJETO = "/content/drive/MyDrive/projeto_deepfake_audio"
    NO_COLAB = True
except ImportError:
    PASTA_PROJETO = "./projeto_deepfake_audio"
    NO_COLAB = False

ARQUIVOS_PARA_CLASSIFICAR = []   # ex.: ["fake/audio.mp3", "original/audio.ogg"]; vazio = abre o upload (Colab)
LIMIAR_DECISAO = None            # None = usa o limiar de operação calibrado no treino (resultados.json)
EXTENSOES = (".wav", ".mp3", ".ogg", ".opus", ".m4a", ".flac", ".aac", ".webm", ".mp4", ".mpeg")

# ---------- pré-processamento e limiar: SEMPRE os do treino (resultados.json) ----------
caminho_cfg = os.path.join(PASTA_PROJETO, "resultados.json")
if not os.path.exists(caminho_cfg):
    raise FileNotFoundError(f"{caminho_cfg} não encontrado: ele guarda o limiar e o pré-processamento do treino.")
with open(caminho_cfg, encoding="utf-8") as _f:
    CFG = json.load(_f)["config"]

# ---------- modelos: todos os do ensemble listados no resultados.json ----------
nomes_modelos = CFG.get("modelos") or ["modelo_cnn_final.keras"]
faltando = [n for n in nomes_modelos if not os.path.exists(os.path.join(PASTA_PROJETO, n))]
if faltando:                                        # nunca classifica com um ensemble incompleto
    raise FileNotFoundError(f"Modelo(s) ausente(s) em {PASTA_PROJETO}: {faltando}. Rode o treino primeiro.")
modelos_classificador = [tf.keras.models.load_model(os.path.join(PASTA_PROJETO, n), compile=False) for n in nomes_modelos]
print("Modelos carregados:", nomes_modelos)
SR, HOP = int(CFG["sr"]), int(CFG["hop"])
N_AMOSTRAS = int(SR * CFG["duracao_s"])
PASSO = int(SR * CFG["passo_janela_s"])
DUR_MIN = 1.0
LIMIAR_DECISAO = float(CFG["limiar_operacao"]) if LIMIAR_DECISAO is None else LIMIAR_DECISAO
MARGEM_INCERTA = float(CFG["margem_incerta"])
print(f"Janela {CFG['duracao_s']} s (passo {CFG['passo_janela_s']} s) | limiar {LIMIAR_DECISAO:.3f} ± {MARGEM_INCERTA}")

def _ffmpeg(args, dados=None):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args], input=dados,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        detalhe = r.stderr.decode(errors="ignore").strip().splitlines()
        raise ValueError(f"arquivo ilegível ou corrompido ({detalhe[-1] if detalhe else 'ffmpeg falhou'})")
    return r.stdout

def _normalizar(y):
    return (0.95 * y / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)

def _mel(t):
    """Idêntico ao treino (seção 1): log-mel normalizado em [0, 1]."""
    S = librosa.feature.melspectrogram(y=t, sr=SR, n_fft=CFG["n_fft"], hop_length=HOP,
                                       n_mels=CFG["n_mels"], fmin=20, fmax=SR // 2)
    return (librosa.power_to_db(S, ref=np.max, top_db=80.0) + 80.0) / 80.0

def _janelas(caminho):
    y = np.frombuffer(_ffmpeg(["-i", caminho, "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"]),
                      dtype=np.float32).copy()
    if len(y) == 0:
        raise ValueError("arquivo sem áudio")
    y, _ = librosa.effects.trim(y, top_db=35)                     # tira silêncio das pontas
    if len(y) < DUR_MIN * SR:
        raise ValueError(f"menos de {DUR_MIN:.0f} s de fala: curto demais para um veredito")
    y = _normalizar(y)
    if len(y) <= N_AMOSTRAS:                                      # curto: repete o áudio
        trechos = [np.tile(y, int(np.ceil(N_AMOSTRAS / len(y))))[:N_AMOSTRAS]]
    else:                                                         # janelas sobrepostas cobrindo o áudio todo
        inicios = list(range(0, len(y) - N_AMOSTRAS + 1, PASSO))
        if inicios[-1] != len(y) - N_AMOSTRAS:
            inicios.append(len(y) - N_AMOSTRAS)
        trechos = [y[i:i + N_AMOSTRAS] for i in inicios]
    return np.stack([_mel(t)[..., None] for t in trechos]).astype(np.float32), len(y) / SR

# ---------- classificação ----------
def classificar_real_ou_fake(caminho, mostrar=True):
    M, dur = _janelas(caminho)
    # ensemble: média das P(fake) dos modelos em cada janela; P(fake) do arquivo = média das janelas
    probs = np.mean([m.predict(M, verbose=0).ravel() for m in modelos_classificador], axis=0)
    p_fake = float(probs.mean())
    if abs(p_fake - LIMIAR_DECISAO) < MARGEM_INCERTA:
        classe = "INCONCLUSIVO"
    else:
        classe = "FAKE" if p_fake >= LIMIAR_DECISAO else "REAL"
    # confiança = distância ao limiar, reescalada para 50–100%
    lado = (1 - LIMIAR_DECISAO) if p_fake >= LIMIAR_DECISAO else LIMIAR_DECISAO
    confianca = 0.5 + 0.5 * min(1.0, abs(p_fake - LIMIAR_DECISAO) / max(lado, 1e-6))
    rotulos = {"FAKE": "🔴 FAKE (voz sintética)", "REAL": "🟢 REAL (voz humana)",
               "INCONCLUSIVO": "🟡 INCONCLUSIVO (perto do limiar)"}
    print(f"\n🎧 {os.path.basename(caminho)}  ({dur:.1f} s de fala, {len(probs)} janela(s))")
    print(f"   ➜ RESULTADO: {rotulos[classe]}  | confiança {100 * confianca:.1f}%"
          f"  | P(fake) = {p_fake:.3f} (limiar {LIMIAR_DECISAO:.2f})")
    print(f"   P(fake) por janela: {[round(float(q), 2) for q in probs]}")
    if classe == "INCONCLUSIVO":
        print("   ⚠️ O modelo está em dúvida. Use um trecho mais longo (10–30 s) de fala contínua.")
    if mostrar:
        fig, ax = plt.subplots(1, 2, figsize=(13, 3), gridspec_kw={"width_ratios": [3, 1]})
        ax[0].imshow(M[0, :, :, 0], aspect="auto", origin="lower", cmap="magma")
        ax[0].set_title(f"{os.path.basename(caminho)} — {classe} ({100 * confianca:.0f}%)")
        ax[0].set_xlabel("tempo (quadros de 10 ms)"); ax[0].set_ylabel("banda mel")
        ax[1].bar(range(len(probs)), probs,
                  color=["tab:red" if q >= LIMIAR_DECISAO else "tab:green" for q in probs])
        ax[1].axhline(LIMIAR_DECISAO, color="k", ls="--")
        ax[1].set_ylim(0, 1); ax[1].set_title("P(fake) por janela"); ax[1].set_xlabel("janela")
        plt.tight_layout(); plt.show()
    return {"arquivo": os.path.basename(caminho), "resultado": classe,
            "confianca_%": round(100 * confianca, 1), "P(fake)": round(p_fake, 3),
            "duracao_s": round(dur, 1), "janelas": len(probs)}

arquivos = list(ARQUIVOS_PARA_CLASSIFICAR)
if not arquivos and NO_COLAB:
    print("Selecione um ou mais áudios para classificar:")
    arquivos = list(files.upload().keys())
if not arquivos:                 # fora do Colab: classifica o que estiver em ./fake e ./original
    arquivos = sorted(os.path.join(p, n) for p in ("fake", "original") if os.path.isdir(p)
                      for n in os.listdir(p) if n.lower().endswith(EXTENSOES))

resumo = []
for caminho in arquivos:
    if not os.path.exists(caminho):
        print(f"❌ {caminho}: arquivo não encontrado"); continue
    try:
        resumo.append(classificar_real_ou_fake(caminho, mostrar=NO_COLAB))
    except Exception as e:
        print(f"❌ {caminho}: não foi possível classificar ({e})")
        resumo.append({"arquivo": os.path.basename(caminho), "resultado": f"ERRO: {e}"})

if resumo:
    print("\n===== RESUMO =====")
    print(pd.DataFrame(resumo).to_string(index=False))
else:
    print("Nenhum áudio classificado. Preencha ARQUIVOS_PARA_CLASSIFICAR ou use o upload.")
