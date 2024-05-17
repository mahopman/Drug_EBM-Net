# Drug EBM-Net
This implementation is built off of the original EBM-Net and the Evidence Inference Dataset.
```
@inproceedings{jin-etal-2020-predicting,
    title = "Predicting Clinical Trial Results by Implicit Evidence Integration",
    author = "Jin, Qiao  and
      Tan, Chuanqi  and
      Chen, Mosha  and
      Liu, Xiaozhong  and
      Huang, Songfang",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.114",
    doi = "10.18653/v1/2020.emnlp-main.114",
    pages = "1461--1477",
    abstract = "Clinical trials provide essential guidance for practicing Evidence-Based Medicine, though often accompanying with unendurable costs and risks. To optimize the design of clinical trials, we introduce a novel Clinical Trial Result Prediction (CTRP) task. In the CTRP framework, a model takes a PICO-formatted clinical trial proposal with its background as input and predicts the result, i.e. how the Intervention group compares with the Comparison group in terms of the measured Outcome in the studied Population. While structured clinical evidence is prohibitively expensive for manual collection, we exploit large-scale unstructured sentences from medical literature that implicitly contain PICOs and results as evidence. Specifically, we pre-train a model to predict the disentangled results from such implicit evidence and fine-tune the model with limited data on the downstream datasets. Experiments on the benchmark Evidence Integration dataset show that the proposed model outperforms the baselines by large margins, e.g., with a 10.7{\%} relative gain over BioBERT in macro-F1. Moreover, the performance improvement is also validated on another dataset composed of clinical trials related to COVID-19.",
}
```

```
@inproceedings{lehman2019inferring,
  title={Inferring Which Medical Treatments Work from Reports of Clinical Trials},
  author={Lehman, Eric and DeYoung, Jay and Barzilay, Regina and Wallace, Byron C},
  booktitle={Proceedings of the North American Chapter of the Association for Computational Linguistics (NAACL)},
  pages={3705--3717},
  year={2019}
}

@misc{deyoung2020evidence,
    title={Evidence Inference 2.0: More Data, Better Models},
    author={Jay DeYoung and Eric Lehman and Ben Nye and Iain J. Marshall and Byron C. Wallace},
    year={2020},
    eprint={2005.04177},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
## Run Drug EBM-Net

### Download Relevant Clinical Trials Information

1. clinical_trials/download_clinical_trials.ipynb

### Prepare Fine-Tuning Dataset

2. evidence_integration/download_evidence_inference.ipynb 
3. evidence_integration/generate_drug_pmcids.ipynb
4. evidence_integration/generate_evidence_integration.ipynb 
5. evidence_integration/index_drug_dataset.ipynb

### Prepare Pre-Training Dataset

6. pretraining_dataset/download_pubmed.ipynb
7. pretraining_dataset/process_pubmed_splits.ipynb
8. pretraining_dataset/generate_drug_pmids.ipynb
9. pretraining_dataset/tag_dataset.ipynb
10. pretraining_dataset/process_tags.ipynb
11. pretraining_dataset/aggregate_contexts.ipynb
12. pretraining_dataset/index_drug_dataset.ipynb

### Run EBM-Net
13. run_drug_ebmnet.ipynb
    Pre-train parameters:
    ```
    model_name_or_path  = f'{local_path}/biobert-v1.1'
    output_dir          = f'{local_path}/pretrained_model'
    do_train            = True
    train_pico          = f'{pretraining_dataset_path}/indexed_evidence.json'
    train_ctx           = f'{pretraining_dataset_path}/indexed_contexts.json'
    num_labels          = 34
    pretraining         = True
    adversarial         = True
    ```
    Fine-tune parameters:
    ```
    model_name_or_path  = f'{local_path}/pretrained_model'
    do_train            = True
    train_pico          = f'{evidence_integration_path}/indexed_train_picos.json'
    train_ctx           = f'{evidence_integration_path}/indexed_train_ctxs.json'
    do_eval             = True
    predict_pico        = f'{evidence_integration_path}/indexed_validation_picos.json'
    predict_ctx         = f'{evidence_integration_path}/indexed_validation_ctxs.json'
    output_dir          = f'{local_path}/ebmnet_model'
    ```

### Run Random EBM-Net
1. evidence_integration/index_random_dataset.ipynb
2. pretraining_dataset/index_random_dataset.ipynb
3. run_drug_ebmnet.ipynb
    Pre-train parameters:
    ```
    model_name_or_path  = f'{local_path}/biobert-v1.1'
    output_dir          = f'{local_path}/random_pretrained_model'
    do_train            = True
    train_pico          = f'{pretraining_dataset_path}/indexed_evidence_random.json'
    train_ctx           = f'{pretraining_dataset_path}/indexed_contexts_random.json'
    num_labels          = 34
    pretraining         = True
    adversarial         = True
    ```
    Fine-tune parameters:
    ```
    model_name_or_path  = f'{local_path}/random_pretrained_model'
    do_train            = True
    train_pico          = f'{evidence_integration_path}/indexed_train_random_picos.json'
    train_ctx           = f'{evidence_integration_path}/indexed_train_random_ctxs.json'
    do_eval             = True
    predict_pico        = f'{evidence_integration_path}/indexed_validation_random_picos.json'
    predict_ctx         = f'{evidence_integration_path}/indexed_validation_random_ctxs.json'
    output_dir          = f'{local_path}/random_ebmnet_model'
    ```

## Run Integration Type Classifier

### Download Relevant Clinical Trials Information

1. clinical_trials/download_clinical_trials.ipynb

### Prepare Data From Evidence Inference
2. evidence_integration/download_evidence_inference.ipynb
3. evidence_integration/generate_drug_pmcids.ipynb

### Prepare Data From PubMed
4. pretraining_dataset/download_pubmed.ipynb
5. pretraining_dataset/process_pubmed_splits.ipynb
6. pretraining_dataset/run generate_drug_pmids.ipynb

### Prepare Intervention Dataset
7. intervention_classifier/generate_intervention_classifier_dataset.ipynb

### Run Intervention Classifier
8. intervention_classifier/intervention_classifier.ipynb
