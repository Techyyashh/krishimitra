---
license: mit
library_name: onnx
pipeline_tag: image-classification
tags: [agriculture, plant-disease, efficientnet, onnx, india]
datasets: [plantvillage, plantdoc]
---

# KrishiMitra – crop leaf disease classifier

Classifies a single leaf photo into **38 classes** (14 crops, healthy + 26 diseases) from the PlantVillage dataset.
Used by the [KrishiMitra app](https://huggingface.co/spaces/YOUR_USERNAME/krishimitra).

## Model
- Architecture: EfficientNet-B0 (timm, ImageNet-pretrained), two-phase transfer learning
- Input: RGB 224×224, ImageNet normalisation (see `classes.json`)
- Output: logits for 38 classes (softmax → probabilities)
- Files: `krishimitra.onnx` (FP32), `krishimitra.int8.onnx` (dynamic INT8), `classes.json`

## Training data
- PlantVillage "color" (~54k lab images, plain backgrounds), stratified 70/15/15 split, seed 42
- Augmentations simulating field conditions: random crops, flips, rotation, brightness/contrast, hue, shadows, blur, noise, JPEG compression
- Class-weighted cross-entropy with label smoothing 0.1

## Evaluation  ✏️ fill in from `reports/*_metrics.json`
| Test set | Images | Accuracy | Macro-F1 | Top-3 |
|---|---|---|---|---|
| PlantVillage test split | – | – | – | – |
| PlantDoc test (real field photos, mapped classes) | – | – | – | – |

CPU latency (ONNX, 1 image): – ms

## Limitations (please read)
- PlantVillage images were taken in a lab on plain backgrounds; **accuracy on real field photos is much lower** (see PlantDoc row).
- Only the 14 crops / 38 classes listed in `classes.json` are known. Other crops will be forced into a wrong class;
  the app shows "not sure" below 60% confidence and rejects non-leaf photos with a colour heuristic.
- Not a substitute for an agricultural expert. Do not use it as the only basis for pesticide decisions.

## Intended use
Decision support and education for farmers and students, with confirmation by local agricultural experts (KVK).
