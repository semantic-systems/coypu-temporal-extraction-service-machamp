# Create JSONLINES Datasets

## Multi-Class Datasets

```
cd jsonlines-conversion-scripts
```

```
python converter_pate.py --input_filepaths ../../original_datasets/pate_and_snips/pate.json \
    --output_directory ../../entity/my_converted_datasets/jsonlines/pate_multi
    --crossvalidation
    --folds 10
```


```
python converter_snips.py --input_filepaths ../../original_datasets/pate_and_snips/snips_train.json \
    ../../original_datasets/pate_and_snips/snips_valid.json
    --output_directory ../../entity/my_converted_datasets/jsonlines/snips_multi
    --crossvalidation
    --folds 10
    --only_temporal
```


```
python converter_fullpate.py \
    --input_filepath_snips ../../entity/jsonlines/snips_multi/snips-full.jsonlines \
    --input_filepath_pate ../../entity/jsonlines/pate_multi/pate-full.jsonlines \
    --output_directory ../../entity/my_converted_datasets/jsonlines/fullpate_multi \
    --crossvalidation \
    --folds 10 \
    --only_temporal
```


```
python converter_wikiwars-tagged.py \
    --input_parent_filepath ../../original_datasets/wikiwars-tagged \
    --output_directory ../../entity/my_converted_datasets/jsonlines/wikiwars-tagged_multi \
    --crossvalidation \
    --folds 10
```


```
python converter_tweets.py \
    --input_test_filepath ../../original_datasets/tweets/testset \
    --input_train_filepath ../../original_datasets/tweets/trainingset/ \
    --output_directory ../../entity/my_converted_datasets/jsonlines/tweets_multi \
    --crossvalidation \
    --folds 10
```


```
python converter_aquaint.py \
    --input_filepaths ../../original_datasets/aquaint \
    --output_directory ../../entity/my_converted_datasets/jsonlines/aquaint_multi \
    --crossvalidation \
    --folds 10
```


```
python converter_timebank.py \
    --input_filepath_timeml ../../original_datasets/timebank/data/timeml \
    --input_filepath_extra ../../original_datasets/timebank/data/extra \
    --output_directory ../../entity/my_converted_datasets/jsonlines/timebank_multi \
    --crossvalidation \
    --folds 10
```


```
python converter_tempeval.py \
    --input_filepath_timebank ../../entity/jsonlines/timebank_multi/timebank-full.jsonlines \
    --input_filepath_aquaint ../../entity/jsonlines/aquaint_multi/aquaint-full.jsonlines \
    --output_directory ../../entity/my_converted_datasets/jsonlines/tempeval_multi \
    --crossvalidation \
    --folds 10
```


## Single-Class Datasets

```
cd jsonlines-conversion-scripts
```

```
python converter_pate.py \
    --input_filepaths ../../original_datasets/pate_and_snips/pate.json \
    --output_directory ../../entity/my_converted_datasets/jsonlines/pate_single \
    --crossvalidation \
    --folds 10 \
    --single_class
```

```
python converter_snips.py \
    --input_filepaths ../../original_datasets/pate_and_snips/snips_train.json ../../original_datasets/pate_and_snips/snips_valid.json \
    --output_directory ../../entity/my_converted_datasets/jsonlines/snips_single \
    --crossvalidation \
    --folds 10 \
    --only_temporal \
    --single_class
```

```
python converter_fullpate.py \
    --input_filepath_snips ../../entity/my_converted_datasets/jsonlines/snips_single/snips-full.jsonlines \
    --input_filepath_pate ../../entity/my_converted_datasets/jsonlines/pate_single/pate-full.jsonlines \
    --output_directory ../../entity/my_converted_datasets/jsonlines/fullpate_single \
    --crossvalidation \
    --folds 10 \
    --only_temporal \
    --single_class
```

```
python converter_wikiwars-tagged.py \
    --input_parent_filepath ../../original_datasets/wikiwars-tagged \
    --output_directory ../../entity/my_converted_datasets/jsonlines/wikiwars-tagged_single \
    --crossvalidation \
    --folds 10 \
    --single_class
```

```
python converter_tweets.py \
    --input_test_filepath ../../original_datasets/tweets/testset \
    --input_train_filepath ../../original_datasets/tweets/trainingset/ \
    --output_directory ../../entity/my_converted_datasets/jsonlines/tweets_single \
    --crossvalidation \
    --folds 10 \
    --single_class
```

```
python converter_aquaint.py \
    --input_filepaths ../../original_datasets/aquaint \
    --output_directory ../../entity/my_converted_datasets/jsonlines/aquaint_single \
    --crossvalidation \
    --folds 10 \
    --single_class
```

```
python converter_timebank.py \
    --input_filepath_timeml ../../original_datasets/timebank/data/timeml \
    --input_filepath_extra ../../original_datasets/timebank/data/extra \
    --output_directory ../../entity/my_converted_datasets/jsonlines/timebank_single \
    --crossvalidation \
    --folds 10 \
    --single_class
```

```
python converter_tempeval.py \
    --input_filepath_timebank ../../entity/my_converted_datasets/jsonlines/timebank_single/timebank-full.jsonlines \
    --input_filepath_aquaint ../../entity/my_converted_datasets/jsonlines/aquaint_single/aquaint-full.jsonlines \
    --output_directory ../../entity/my_converted_datasets/jsonlines/tempeval_single \
    --crossvalidation \
    --folds 10 \
    --single_class
```




# References

* [UzZaman et al., 2013] [UzZaman, N., Llorens, H., Derczynski, L., Allen, J., Verhagen, M., and Pustejovsky, J. (2013). Semeval-2013 task 1: Tempeval-3: Evaluating time expressions, events, and temporal relations. In Second Joint Conference on Lexical and Computational Semantics (* SEM), Volume 2: Proceedings of the Seventh International Workshop on Semantic Evaluation (SemEval 2013), pages 1–9. ](https://aclanthology.org/S13-2001.pdf)

* [Derczynski et al., 2012] [Derczynski, L., Llorens, H., and Saquete, E. (2012). Massively increasing timex3 resources: a transduction approach. arXiv preprint arXiv:1203.5076.](https://arxiv.org/abs/1203.5076)

* [Mazur and Dale, 2010] [Mazur, P. and Dale, R. (2010). Wikiwars: A new corpus for research on temporal expressions. In Proceedings of the 2010 conference on empirical methods in natural language processing, pages 913–922](https://aclanthology.org/D10-1089.pdf)

* [Zhong et al., 2017] [Zhong, X., Sun, A., and Cambria, E. (2017). Time expression analysis and recognition using syntactic token types and general heuristic rules. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 420–429, Vancouver, Canada. Association for Computational Linguistics.](https://aclanthology.org/P17-1039/)

* [Zarcone et al., 2020] [Zarcone, A., Alam, T., and Kolagar, Z. (2020). Pâté: a corpus of temporal expressions for the in-car voice assistant domain. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 523–530.](https://aclanthology.org/2020.lrec-1.66/)