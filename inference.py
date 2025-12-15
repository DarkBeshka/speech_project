import torch
from TTS.utils.synthesizer import Synthesizer

cfg_path = r"ruslan_glowtts_exp\run-December-11-2025_12+54PM-0000000\config.json"
model_path = r"ruslan_glowtts_exp\run-December-11-2025_09+56AM-0000000\best_model_131850.pth"

#cоздаем синтезатор, будем использовать Griffin-Lim
synth = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=cfg_path,
    use_cuda=torch.cuda.is_available(),
)

wav = synth.tts("привет! это тест модели")
synth.save_wav(wav, "output.wav")