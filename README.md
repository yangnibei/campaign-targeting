# Campaign Targeting

**Yang Nibei · Business Analytics Portfolio · 04 / 10**

> Can a pre-contact score prioritise a limited call budget?

[Portfolio](https://github.com/yangnibei) · [Results & decisions](REPORT.md) · [Python analysis](analysis.py)

![Campaign Targeting: reproducible analysis](analysis.svg)

## Visual guide

- **Left — cumulative capture:** shows how much observed response is captured as the contact budget expands, with the 20% example marked.
- **Right — lift by budget:** shows ranking quality relative to the held-out population; it is response propensity, not causal uplift.

## Business brief

This public-data case study explores a focused business question using an **order-preserving last-block holdout, logistic regression, precision and lift**. Read the results alongside their assumptions before acting on them.

## Approach

Order-preserving last-block holdout, logistic regression, precision and lift. The dataset lacks a complete timestamp, so this is not described as a strict chronological split. The implementation contains input and reconciliation assertions. Each run writes an aggregate report, machine-readable metrics and the chart shown above.

## Reproduce

Python 3.11+ recommended. From this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v
python analysis.py
```

On Windows, activate with `.venv\\Scripts\\activate`. Use `--data-dir /absolute/path/to/cache` and `--output-dir /absolute/path/to/results` to keep downloads and regenerated outputs elsewhere. The first public-data run requires internet access. Raw data stays in the ignored `data/` directory; it is not redistributed in this repository. Runs reuse cached archives and record their SHA-256 hashes in `results.json`.

## Data & attribution

UCI Machine Learning Repository, dataset key `bank`; exact citation, download URL, licence and source hash are recorded in [results.json](results.json) after execution. Original data is CC BY 4.0; attribution is preserved. See `SOURCES` in [utils.py](utils.py). These are historical teaching datasets, not current business measurements.

## Limitations

Offline response ranking is not causal marketing uplift. No contact-duration information or protected demographic fields are used.

This repository is an educational portfolio case study, not paid client work, employment evidence or a production system. Code and documentation were developed with AI assistance and executed against the stated data; that does not imply independent third-party validation.

## Repository guide

- `analysis.py`: project-specific calculations and checks.
- `utils.py`: download, provenance and reporting helpers.
- `test_analysis.py`: offline helper unit tests.
- `REPORT.md`: generated findings and decision boundaries.
- `results.json`: aggregate metrics and source provenance.
- `analysis.svg`: reproducible figure.
