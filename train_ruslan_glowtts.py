import os
import sys
from trainer import Trainer, TrainerArgs
import trainer.trainer as trainer_module
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

_venv_sp = os.path.join(os.path.dirname(__file__), ".venv311", "Lib", "site-packages")
if os.path.isdir(_venv_sp) and _venv_sp not in sys.path:
    sys.path.insert(0, _venv_sp)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True") #снижение фрагментации CUDA-памяти

OUTPUT_PATH = os.path.abspath("ruslan_glowtts_exp")
DATASET_PATH = os.path.abspath("data_22050")

os.makedirs(OUTPUT_PATH, exist_ok=True)

#конфиг датасета(ruslan formatter)
dataset_config = BaseDatasetConfig(
    formatter="ruslan",
    meta_file_train="metadata_train.txt", #скрипт, разделяющий на train/val лежат в папке с датасетом
    meta_file_val="metadata_val.txt",
    path=DATASET_PATH,
    language="ru",
)

#обработка блокировки Windows(из-за нее падало обучение после каждой эпохи)
_orig_remove_experiment_folder = trainer_module.remove_experiment_folder

def _safe_remove_experiment_folder(path):
    try:
        _orig_remove_experiment_folder(path)
    except PermissionError as exc:
        print(f"[warn] Не удалось удалить {path} из-за блокировки: {exc}. Пропускаю очистку.")

trainer_module.remove_experiment_folder = _safe_remove_experiment_folder


#конфиг модели GlowTTS
config = GlowTTSConfig(
    batch_size=8,          # уменьшено для экономии GPU-памяти
    eval_batch_size=8,
    num_loader_workers=2,
    num_eval_loader_workers=2,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=32,
    use_phonemes=False,
    text_cleaner="basic_cleaners",
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    output_path=OUTPUT_PATH,
    datasets=[dataset_config],
    #попытка исправитье взрывающиеся градиенты:
    lr=0.0002,              #уменьшаем learning rate (было 0.001 по умолчанию)
    grad_clip=1.0,          #больший gradient clipping (было 5.0)
    lr_scheduler="NoamLR",
    lr_scheduler_params={
        "warmup_steps": 8000,  #увеличиваем warmup для более плавного старта
    },
)

config.characters = CharactersConfig(
    pad="<PAD>",
    eos="<EOS>",
    bos="<BOS>",
    blank="<BLNK>",
    characters="абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
               "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ",
    punctuations=" !'(),-.:;?«»”/*…–—’“„<>",
    phonemes=None,
    is_unique=True,
    is_sorted=False,
)

#аудио-процессор
ap = AudioProcessor.init_from_config(config)

#токенизатор(символьный)
tokenizer, config = TTSTokenizer.init_from_config(config)

#загружаем семплы (список [text, audio_path, speaker_name])
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=False,  #мы явно указали meta_file_val
)

def _fix_sample_paths(samples, dataset_root):
    """Исправляет пути до аудио(костыль)"""
    if not samples:
        return 0, 0
    fixed = 0
    checked = 0

    common_keys = [
        "audio",
        "audio_path",
        "audio_filepath",
        "path",
        "wav",
        "audio_file",
        "file",
        "filepath",
    ]

    for i, s in enumerate(samples):
        try:
            audio_p = None
            setter = None

            if isinstance(s, (list, tuple)):
                if len(s) < 2:
                    continue
                checked += 1
                audio_p = s[1]

                def _set_list_path(new_p, idx=i, samples_ref=samples):
                    cur = samples_ref[idx]
                    if isinstance(cur, list):
                        cur[1] = new_p
                    else:
                        newt = tuple([cur[0], new_p] + list(cur[2:]))
                        samples_ref[idx] = newt

                setter = _set_list_path

            elif isinstance(s, dict):
                key = next((k for k in common_keys if k in s), None)
                if key is None:
                    if 1 in s:
                        key = 1
                if key is None:
                    continue
                checked += 1
                audio_p = s.get(key)

                def _set_dict_path(new_p, dict_ref=s, k=key):
                    dict_ref[k] = new_p

                setter = _set_dict_path

            else:
                continue

            if not audio_p:
                continue

            if os.path.isabs(audio_p):
                if os.path.exists(audio_p):
                    continue
            else:
                if os.path.exists(audio_p):
                    continue

            base_candidate = os.path.join(dataset_root, os.path.basename(str(audio_p)))
            if os.path.exists(base_candidate):
                setter(base_candidate)
                fixed += 1
                continue

            joined_candidate = os.path.join(dataset_root, str(audio_p))
            if os.path.exists(joined_candidate):
                setter(joined_candidate)
                fixed += 1
                continue

        except Exception:
            print("Warning: failed to inspect/fix sample at index", i, "error:", sys.exc_info()[1])
            continue

    return checked, fixed

checked_train, fixed_train = _fix_sample_paths(train_samples, DATASET_PATH)
checked_eval, fixed_eval = _fix_sample_paths(eval_samples, DATASET_PATH)
print(f"Fixed audio paths: train checked={checked_train} fixed={fixed_train}, eval checked={checked_eval} fixed={fixed_eval}")

#если валидационные сэмплы отсутствуют/None, отключаем eval, чтобы не падало обучение
if not eval_samples:
    eval_samples = []
    config.run_eval = False
    print("[warn] eval_samples пусты, отключаю run_eval.")

#сама модель
model = GlowTTS(config, ap, tokenizer, speaker_manager=None) #один спикер → speaker_manager=None

#трейнер
trainer = Trainer(
    TrainerArgs(),
    config,
    OUTPUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

if __name__ == "__main__":
    trainer.fit()
