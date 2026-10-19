"""Publish the official runtime without changing existing WeatherPre Spaces."""
from pathlib import Path
import shutil
import tempfile
from huggingface_hub import HfApi


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as folder:
        dest = Path(folder)
        for name in ('weatherai', 'scripts'):
            shutil.copytree(root/name, dest/'WeatherAI'/name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for file in (root/'hf_official').iterdir():
            if file.is_file():
                shutil.copy2(file, dest/file.name)
        print(HfApi().upload_folder(repo_id='LonghaoWang/weatherai-official', repo_type='space',
                                   folder_path=folder, commit_message='Use pinned author code and published checkpoints'))


if __name__ == '__main__':
    main()
