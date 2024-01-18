---
library_name: setfit
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
metrics:
- accuracy
widget:
- text: 'this is a story of two misfits who do n''t stand a chance alone , but together
    they are magnificent . '
- text: 'it does n''t believe in itself , it has no sense of humor ... it ''s just
    plain bored . '
- text: 'the band ''s courage in the face of official repression is inspiring , especially
    for aging hippies ( this one included ) . '
- text: 'a fast , funny , highly enjoyable movie . '
- text: 'the movie achieves as great an impact by keeping these thoughts hidden as
    ... ( quills ) did by showing them . '
pipeline_tag: text-classification
inference: true
base_model: sentence-transformers/paraphrase-mpnet-base-v2
model-index:
- name: SetFit with sentence-transformers/paraphrase-mpnet-base-v2
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: Unknown
      type: unknown
      split: test
    metrics:
    - type: accuracy
      value: 0.8588082901554405
      name: Accuracy
---

# SetFit with sentence-transformers/paraphrase-mpnet-base-v2

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. This SetFit model uses [sentence-transformers/paraphrase-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [sentence-transformers/paraphrase-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 512 tokens
- **Number of Classes:** 2 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label    | Examples                                                                                                                                                                                                                                                                                         |
|:---------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| negative | <ul><li>'stale and uninspired . '</li><li>"the film 's considered approach to its subject matter is too calm and thoughtful for agitprop , and the thinness of its characterizations makes it a failure as straight drama . ' "</li><li>"that their charm does n't do a load of good "</li></ul> |
| positive | <ul><li>"broomfield is energized by volletta wallace 's maternal fury , her fearlessness "</li><li>'flawless '</li><li>'insightfully written , delicately performed '</li></ul>                                                                                                                  |

## Evaluation

### Metrics
| Label   | Accuracy |
|:--------|:---------|
| **all** | 0.8588   |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("a fast , funny , highly enjoyable movie . ")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median  | Max |
|:-------------|:----|:--------|:----|
| Word count   | 2   | 11.4375 | 33  |

| Label    | Training Sample Count |
|:---------|:----------------------|
| negative | 8                     |
| positive | 8                     |

### Training Hyperparameters
- batch_size: (16, 16)
- num_epochs: (4, 4)
- max_steps: -1
- sampling_strategy: oversampling
- body_learning_rate: (2e-05, 1e-05)
- head_learning_rate: 0.01
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: True

### Training Results
| Epoch   | Step   | Training Loss | Validation Loss |
|:-------:|:------:|:-------------:|:---------------:|
| 0.1111  | 1      | 0.2116        | -               |
| 1.0     | 9      | -             | 0.2229          |
| 2.0     | 18     | -             | 0.1815          |
| **3.0** | **27** | **-**         | **0.1729**      |
| 4.0     | 36     | -             | 0.1752          |

* The bold row denotes the saved checkpoint.
### Framework Versions
- Python: 3.10.13
- SetFit: 1.0.3
- Sentence Transformers: 2.2.2
- Transformers: 4.36.2
- PyTorch: 2.1.2+cu121
- Datasets: 2.16.1
- Tokenizers: 0.15.0

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->