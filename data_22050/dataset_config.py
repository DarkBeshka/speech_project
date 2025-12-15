from TTS.tts.configs.shared_configs import BaseDatasetConfig

dataset_config = BaseDatasetConfig(
    formatter="ruslan",                # встроенный форматтер
    meta_file_train="metadata_train.txt",
    meta_file_val="metadata_val.txt",
    path="/path/to/data/ruslan",       # путь к ruslan_root
    language="ru"                      # язык датасета
)