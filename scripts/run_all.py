"""Run the revised analysis from any working directory."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures', action='store_true',
                        help='also regenerate survey, clustering, and model PNGs at 600 dpi')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    scripts = ['run_structure.py', 'run_cluster_checks.py', 'run_models.py',
               'draw_pca_figure.py', 'check_against_manuscript.py']
    if args.figures:
        scripts += ['draw_survey_figures.py', 'draw_cluster_figure.py',
                    'draw_model_figures.py']
    for name in scripts:
        print(f'Running {name}', flush=True)
        subprocess.run([sys.executable, str(root / 'scripts' / name)],
                       cwd=root, check=True)


if __name__ == '__main__':
    main()
