# RainHash — Experimental Results

This directory contains the experimental datasets and supporting files associated with the paper **"RainHash over Fp via Quadratic Extension Towers with Algebraic and Cryptanalytic Analysis."**

The experiments evaluate the statistical behavior and diffusion properties of the proposed RainHash construction over the finite field F_251, using the parameter α = 4 and eight permutation rounds.

## Experimental Archives

| Archive                                         | Description                                                                                                                       | Paper   |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `resultats_independance_alpha4.zip`             | Output-coordinate uniformity, pairwise independence, coordinate sums and differences, synthetic controls, and validation results. | Table 2 |
| `resultats_differentiel_alpha4_complet.zip`     | First- and second-order differential experiments, exploratory statistics, independent validation, and Monte Carlo testing.        | Table 3 |
| `resultats_avalanche_alpha4_272000_complet.zip` | Avalanche evaluation over 272,000 message pairs, including coordinate-change distributions, binomial tests, and Holm corrections. | Table 4 |
| `resultats_projections_alpha4_complet.zip`      | Linear-projection experiments, affine and quadratic relation diagnostics, and confirmatory statistical testing.                   | Table 5 |
| `rapport_tours_alpha4.zip`                      | Round-by-round diffusion analysis over eight rounds, including uniform and structured input states.                               | Table 6 |
| &#x20;                                          | Generated binary sequences, extraction controls, and metadata associated with the NIST STS and TestU01 experiments.               | Table 7 |

## Data Organization

The archives contain experimental outputs in formats including:

- **CSV:** Numerical measurements and statistical results.
- **JSON:** Experimental configurations, summaries, and metadata.
- **NPY/NPZ:** Saved numerical arrays and intermediate computational states.
- **BIN:** Binary sequences generated for randomness testing.
- **Python:** Source files included in selected archives.

The main implementation and experimental scripts are available in the root directory of this repository.

## Reproducibility

The experimental configuration, numerical results, and supporting data are provided to facilitate independent inspection and reproducibility.

Each archive can be downloaded and extracted separately.

The NIST STS and TestU01 archive contains the generated binary streams and extraction controls. Complete per-test battery reports are not included in the current archive.




