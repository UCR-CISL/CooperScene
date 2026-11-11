"""Download CooperScene configs and checkpoints from the Hugging Face model zoo.

Examples:
    python tools/download.py ermvp                   # one baseline
    python tools/download.py ermvp cosdh             # several
    python tools/download.py bevfusion_coop_lidar    # one BEVFusion variant
    python tools/download.py bevfusion               # all BEVFusion variants
    python tools/download.py all                     # the whole model zoo

Files land in ``assets/<model>/{<model>.py, <model>.pth}`` (or ``--local-dir``),
ready for ``tools/test.py`` and ``tools/train.py``.
"""
import argparse
import sys

from huggingface_hub import HfApi, snapshot_download

REPO_ID = 'cisl-hf/CooperScene'
# Shared files every model needs (configs inherit from _base_).
COMMON = ['config.json', '_base_/*']


def available_models(files):
    """Map each downloadable name to its file pattern in the repo."""
    models = {}
    for path in files:
        if not path.endswith('.pth'):
            continue
        folder, stem = path.rsplit('/', 1)[0], path.rsplit('/', 1)[1][:-4]
        models[stem] = f'{folder}/{stem}.*'
        models.setdefault(folder, f'{folder}/*')
    return models


def main():
    parser = argparse.ArgumentParser(
        description='Download CooperScene configs + checkpoints.')
    parser.add_argument(
        'models', nargs='+',
        help="model name(s), e.g. 'ermvp', 'bevfusion_coop_lidar', or 'all'")
    parser.add_argument(
        '--local-dir', default='assets',
        help='destination directory (default: assets)')
    parser.add_argument(
        '--revision', default='main', help='model zoo revision')
    args = parser.parse_args()

    files = HfApi().list_repo_files(REPO_ID, revision=args.revision)
    models = available_models(files)

    if 'all' in args.models:
        patterns = None  # everything
    else:
        unknown = [m for m in args.models if m not in models]
        if unknown:
            sys.exit(f"Unknown model(s): {', '.join(unknown)}\n"
                     f"Available: all, {', '.join(sorted(models))}")
        patterns = COMMON + [models[m] for m in args.models]

    path = snapshot_download(
        REPO_ID, revision=args.revision, local_dir=args.local_dir,
        allow_patterns=patterns)
    print(f'Downloaded to {path}')


if __name__ == '__main__':
    main()
