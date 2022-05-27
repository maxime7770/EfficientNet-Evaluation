# Evaluation of the EfficientNet model

This project aims at evaluating the performances of the EfficientNet-b0 model while building an inference pipeline and analyzing the performances using different metrics.

## Dataset

For that task, we use a 50,000 images dataset. Images are equally distributed among 1000 classes.


## Results

Here are the global results (on the entire dataset and for all classes):

| Metrics | Values |
| :----- | :----- |
|  Accuracy       |  74.3% |
|  Top 5 Accuracy |  91.9% |
|  F1 Score       |  74.3% |
|  Precision      |  74.3% |
|  Recall         |  74.3% |
|  Specificity    | 99.97% |


Please check the `main.ipynb` and `classification_report.csv` files for more results such as metrics' distributions per classes, analysis, interpretations and more.


