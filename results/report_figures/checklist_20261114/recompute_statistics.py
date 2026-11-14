"""Recompute paired, image-level statistics for the fixed official test CSVs.

Run from the repository root. No training, checkpoint selection or threshold tuning.
Requires numpy, pandas, scipy. Outputs are written next to this script.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def main():
    tables, hashes = {}, {}
    for name in 'ABC':
        path = ROOT / f'results/{name}_test/per_image.csv'
        table = pd.read_csv(path, dtype={'image_id': str})
        assert len(table) == 150 and table.image_id.is_unique
        assert table['split'].eq('test').all()
        assert np.isfinite(table.dice).all()
        tables[name] = table[['image_id', 'dice']]
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []
    for before, after in [('A', 'B'), ('B', 'C'), ('A', 'C')]:
        assert set(tables[before].image_id) == set(tables[after].image_id)
        paired = tables[before].merge(tables[after], on='image_id', validate='one_to_one')
        delta = (paired.dice_y - paired.dice_x).to_numpy()
        rng = np.random.default_rng(0)
        bootstrap = delta[rng.integers(0, 150, size=(10000, 150))].mean(axis=1)
        low, high = np.percentile(bootstrap, [2.5, 97.5])
        rows.append(dict(comparison=f'{after}_vs_{before}', delta=float(delta.mean()),
                         ci_low=float(low), ci_high=float(high),
                         p_wilcoxon=float(wilcoxon(delta, alternative='two-sided').pvalue),
                         images_up=int((delta > 0).sum()), images_down=int((delta < 0).sum()),
                         images_equal=int((delta == 0).sum())))
    result = dict(bootstrap_samples=10000, bootstrap_seed=0, ci='percentile 2.5/97.5',
                  wilcoxon='paired, two-sided, scipy defaults; no multiplicity adjustment',
                  source_sha256=hashes, comparisons=rows,
                  scope='Conditional on fixed checkpoints and 150 test images. '
                        'Not uncertainty over seeds/training runs. '
                        'Patient/video independence remains unverified.')
    (OUT / 'paired_statistics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    pd.DataFrame(rows).to_csv(OUT / 'paired_statistics.csv', index=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
