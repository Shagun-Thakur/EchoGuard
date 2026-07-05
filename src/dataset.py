from pathlib import Path

def get_audio_files(folder):
    return sorted(Path(folder).glob("*.wav"))

def dataset_summary(normal_files, abnormal_files):
    return{
        "normal": len(normal_files),
        "abnormal": len(abnormal_files)
    }
