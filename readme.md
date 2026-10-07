# Few-Shot Chest X-Ray Classification with Prototypical Networks

A research and demonstration project for classifying chest X-ray images into five selected categories using a Prototypical Network (ProtoNet) and a ResNet50-based embedding model.

> **Research/demo use only.** This project is not a medical device and must not be used to diagnose, treat, or rule out disease.

## Overview

The project focuses on few-shot classification under class imbalance. A ResNet50-based model maps an X-ray image to a 126-dimensional embedding. During episodic evaluation, class prototypes are calculated by averaging the embeddings of support images, and query images are assigned to the nearest prototype using squared Euclidean distance.

The five selected classes are:

- Hernia
- Pneumonia
- Fibrosis
- Edema
- Emphysema

## Method

1. Resize each image to `224 × 224 × 3`.
2. Scale pixel values to `[0, 1]` before passing them into the saved model. The model's internal preprocessing layer handles ResNet50 preprocessing.
3. Extract 126-dimensional embeddings.
4. Average support embeddings within each class to construct prototypes.
5. Compute squared Euclidean distances from the query embedding to each prototype.
6. Predict the class with the smallest distance.

For class \(c\), the prototype is

\[
p_c = \frac{1}{K}\sum_{i=1}^{K} f_\theta(x_i),
\]

where \(K\) is the number of support examples per class and \(f_\theta\) is the embedding model.

The predicted class for a query \(x\) is

\[
\hat{y} = \arg\min_c \|f_\theta(x)-p_c\|_2^2.
\]

## Dataset

The project uses the NIH Chest X-ray dataset from the Hugging Face repository [`alkzar90/NIH-Chest-X-ray-dataset`](https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset), selecting five categories.

The recorded split sizes are:

| Split | Records |
|---|---:|
| Training | 1,600 |
| Validation | 358 |
| Test | 350 |

The recorded test-set class counts are:

| Class | Test images |
|---|---:|
| Hernia | 15 |
| Pneumonia | 35 |
| Fibrosis | 155 |
| Edema | 44 |
| Emphysema | 101 |
| **Total** | **350** |

The project uses a patient-level split to avoid patients appearing across multiple partitions. Confirm this split in your own data-preparation pipeline before reporting results.

## Evaluation

The research evaluation uses a 5-way 5-shot episodic protocol. The recorded comparison across 100 episodes (2,500 query predictions) is:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Base CNN | 27.96% | 27.75% |
| ResNet50 embedding baseline | 43.12% | 41.98% |
| ProtoNet | 44.08% | 43.13% |

These results indicate a modest improvement for ProtoNet over the ResNet50 embedding baseline under the recorded evaluation setup. They should not be interpreted as evidence of clinical effectiveness. For reproducibility, ensure all models are evaluated using the same episode-generation protocol and, ideally, the same sampled episodes.

## Gradio Demo

The Gradio demo accepts an uploaded X-ray and compares its embedding against five class prototypes created from a fixed 5-shot support set (25 support images total).

### Demo behavior

- Five support images are selected for each class.
- Support embeddings are averaged to create five prototypes.
- The uploaded image is embedded and compared with each prototype.
- The predicted class is the one with the smallest squared Euclidean distance.
- The UI displays the predicted class and the distance to each prototype.

The fixed support set should be selected once and kept unchanged between predictions. The prototype distances are similarity measures in the learned embedding space, **not calibrated probabilities**.

### Important data-split note

If the fixed support images are selected from the test partition, those images are being used for the demo's support set and should not also be treated as independent test queries in a final benchmark. For a cleaner deployment workflow, choose the support set from a designated support/training partition and reserve the test set for evaluation.

## Requirements

Suggested core dependencies:

- Python 3.10 or a compatible version supported by your TensorFlow installation
- TensorFlow
- NumPy
- Gradio
- Pandas (for preparing the path/label tables)

Install the Python packages in your environment, for example:

```bash
pip install tensorflow numpy gradio pandas
```

Use versions compatible with the environment in which the model was trained and saved.

## Project Setup

1. Clone or download this repository.
2. Install the required dependencies.
3. Obtain the dataset and prepare patient-level training, validation, and test splits.
4. Ensure your data table contains the image path and class label for each record.
5. Place the saved Keras model at the configured path, or update `MODEL_PATH` in the notebook/script.
6. Build the class-specific image-path lists.
7. Select the fixed support set, load its images, compute the embeddings, and construct the prototypes.
8. Start the Gradio app.

Example model path used in the notebook:

```text
/workspace/DeepLearningProject/saved_models/protonet_model.keras
```

Update this path to match your environment.

## Launching Gradio

Run the notebook cell or Python script containing the Gradio interface:

```python
demo.launch(
    server_name="0.0.0.0",
    server_port=6007,
    share=False
)
```

Open the app at:

```text
http://localhost:6007
```

If running on a remote notebook or server, port forwarding or an approved public-sharing option may be required. `server_name="0.0.0.0"` makes the app listen on available interfaces; it does not by itself make the app publicly accessible.

## Model and Data Notes

- The embedding layer used by the demo is expected to produce vectors of shape `(batch_size, 126)`.
- The prototype tensor is expected to have shape `(5, 126)`.
- The input image is resized to `224 × 224` and scaled to `[0, 1]`.
- Do not apply ResNet50 `preprocess_input` a second time outside the model if the saved model already contains that preprocessing operation.
- Ensure each class has at least five available support images.
- File decoding must match the actual image format (for example, PNG versus JPEG).
- Save the selected support paths or random seed if you need to reproduce the exact demo prototypes.

## Limitations

- Only five selected categories are covered; this is not a general-purpose chest X-ray classifier.
- The support set affects prototype positions and predictions.
- The recorded test set is imbalanced, including only 15 Hernia images.
- Reported results are experimental and require further validation, uncertainty estimates, and careful statistical analysis.
- Prediction stability across support sets is not the same as model confidence or clinical probability.
- The project does not replace professional radiological assessment.

## Citation

Dataset:

> NIH Chest X-ray dataset, distributed via the Hugging Face repository [`alkzar90/NIH-Chest-X-ray-dataset`](https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset).

Prototypical Networks:

> Snell, J., Swersky, K., and Zemel, R. “Prototypical Networks for Few-shot Learning.” *Advances in Neural Information Processing Systems*, 2017.

ResNet:

> He, K., Zhang, X., Ren, S., and Sun, J. “Deep Residual Learning for Image Recognition.” *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2016.

## License

Add the license that applies to your code and verify the dataset's license and terms before redistributing data or derived artifacts.
