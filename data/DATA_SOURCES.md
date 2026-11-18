# Data sources

Raw data is **not** stored in Git (see `.gitignore`). Download it with `notebooks/train_on_colab.ipynb`.

| Dataset | Use | Where | License / terms |
|---|---|---|---|
| PlantVillage (color) – 38 classes, ~54k images | Training, validation, test | Kaggle: `abdallahalidev/plantvillage-dataset` (original: github.com/spMohanty/PlantVillage-Dataset) | Check the license on the Kaggle page and original repository before redistribution. Cite Hughes & Salathé (2015). |
| PlantDoc – real field photos | Robustness test (and optional fine-tuning) | github.com/pratikkayal/PlantDoc-Dataset | CC BY 4.0 (verify in repo). Cite Singh et al. (2020). |

## Citation
- D. P. Hughes, M. Salathé. *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv:1511.08060, 2015.
- D. Singh et al. *PlantDoc: A Dataset for Visual Plant Disease Detection.* CoDS-COMAD 2020.

## Data leakage note
PlantVillage contains several photos of the same leaf from different angles. A random split can put
near-duplicates in both train and test, which makes test accuracy look better than real life.
This is one more reason to also report accuracy on PlantDoc.
