# Overview

This documentation describes how to work with this directory.
In general, it is recommended to prepare the data first, before working with the [UIE](../uie) and [MaChAmp](../machamp) directories to work with the models.
It is also recommended to work with the conventions set by this documentation.
In particular, it is important that the freshly converted datasets are saved in the correct directory to make the use of all the scripts in this repository easy.

This directory contains already converted datasets in most of the required formats.
It is possible to use the pre-converted datasets with the models (both UIE and MaChAmp).
If the whole crossvalidation process is to be reproduced, it is required to convert the datasets with the provided scripts, because the crossvalidation folds are not available in this directory (they can be easily produced with the provided scripts).

The structure of this directory is the following:

```text
temporal-data
├── ..                      # Main directory
├── entity                  # Contains all the converted datasets used in the thesis
├── original_datasets       # Contains the original versions of the datasets
├── relation                # Contains the temporal relation extraction datasets in UIE-format
├── scripts                 # Contains all the scripts to convert the data to any format
```

To use the scripts in the [scripts directory](scripts) it is required that the required Python packages are downloaded.
For this it is recommended to setup Anaconda as described in the [main directory](..).
The scripts run with either the MaChAmp or the UIE environment.
For each of the original datasets a converter script has been written, that converts the data to a JSONLINE format (these are in the [jsonlines conversion directory](scripts/jsonlines-conversion-scripts/)).
This format can not directly be used by either UIE or MAChAmp, but further scripts can use this format (for example [dataset analysis scripts](scripts/dataset-analysis-scripts/) that calculate the statistics for each dataset).
Furthermore, it is important to first download the [nltk punkt tokenizer](https://www.nltk.org/api/nltk.tokenize.punkt.html).
The [scripts directory](scripts) contains a Python script that downloads the punkt tokenizer automatically.





# Temporal Datasets

There are four datasets used by the thesis: Fullpate, TempEval-3, WikiWars and Tweets.
The Fullpate dataset is originally called PATE [[Zarcone et al., 2020]](#References), but in the context of this repository the name Fullpate is used.
The reason for this is that the Fullpate dataset consists of two pieces: Snips and Pate.
These pieces have a different origin and format structure.
In terms of temporal semantics these datasets are quite different.
Still, both of the pieces label temporal expressions with one of the four temporal classes: date, time, duration, set.
Similarly, the TempEval-3 dataset [[UzZaman et al., 2013]](#references) consists of two pieces, namely TimeBank and AQUAINT.

The other two datasets are Tweets [[Zhong et al., 2017]](#references) and WikiWars [[Mazur and Dale, 2010]](#references).
WikiWars originally was only TIMEX2 tagged i.e. it does not contain any information on temporal classes.
In the work by Derczynski et al. [[Derczynski et al., 2012]](#references), the dataset was labeled with TIMEX3 temporal classes.

Since TempEval-3 and Fullpate consists of subsets which have a different format, there are converter scripts for each of the subsets as well as their union.
The original datasets can be found in the [original_datasets directory](original_datasets). 





# Data format

There are three target formats, which are used in this repository: JSONLINES, UIE and BIO.
Furthermore, each of the original datasets has its own format.
Most of them follow an XML structure, despite the PATE dataset [[Zarcone et al., 2020]](#References), which consists of two different JSON formats.

[![Temporal Conversion Formats Overview](../docs/images/temporal-conversion-formats.png)](#temporal-datasets)
> The graphic shows the datasets, formats and the relations between them.

## JSONLINES format


## MaChAmp format (BIO)

The original datasets are mostly in an XML format with inline tagged TIMEX3 tags, but some are in a JSON format.
UIE uses a special json format that contains both SSI and SEL per entry.
The MaChAmp models use a multiclass BIO format with 9 different labels:
* B-DATE      
* B-TIME      
* B-DURATION  
* B-SET       
* I-DATE      
* I-TIME      
* I-DURATION  
* I-SET       
* O      


## UIE format

The UIE models use a JSON format with four temporal classes: date, time, duration, set.
Similar to T5 (text-to-text) models, each data-entry consists of a prompt called "SSI" and target graphlike structure called "SEL". 




# Conversion scripts






# Original datasets sources

All original datasets can be found on the internet, but they are also uploaded in the [original_datasets directory](original_datasets).
For example, the following links can be used for download:

* [Fullpate (Pate and Snips)](https://zenodo.org/records/3697930#.ZBwzbi00hQI):
* [WikiWars (tagged)](https://github.com/satya77/Transformer_Temporal_Tagger/blob/master/data.zip)
* [Tweets](https://github.com/xszhong/syntime/tree/master/syntime/resources/tweets)
* [TimeBank + AQUAINT](https://github.com/satya77/Transformer_Temporal_Tagger/blob/master/data.zip)





# How to add new datasets?

To use either MaChAmp or UIE on new datasets it is recommended to write a converter script similar to the ones in the [jsonlines converter directory](scripts/jsonlines-conversion-scripts/).
After that, the [BIO converter](scripts/bio-conversion-scripts/) can be used to convert to the MaChAmp fromat or the [UIE converter](scripts/uie-conversion-scripts/) to convert to the UIE format.
Furthermore, UIE requires the generation of a [YAML configuration file](scripts/uie-conversion-scripts/data_config/entity/) and another dataset specific [converter module](scripts/uie-conversion-scripts/universal_ie/task_format/).
The [converter module directory](scripts/uie-conversion-scripts/universal_ie/task_format/) contains example scripts (e.g. [fullpate.py](scripts/uie-conversion-scripts/universal_ie/task_format/fullpate.py) or [tweets.py](scripts/uie-conversion-scripts/universal_ie/task_format/tweets.py)).





# References

* [UzZaman et al., 2013] [UzZaman, N., Llorens, H., Derczynski, L., Allen, J., Verhagen, M., and Pustejovsky, J. (2013). Semeval-2013 task 1: Tempeval-3: Evaluating time expressions, events, and temporal relations. In Second Joint Conference on Lexical and Computational Semantics (* SEM), Volume 2: Proceedings of the Seventh International Workshop on Semantic Evaluation (SemEval 2013), pages 1–9. ](https://aclanthology.org/S13-2001.pdf)

* [Derczynski et al., 2012] [Derczynski, L., Llorens, H., and Saquete, E. (2012). Massively increasing timex3 resources: a transduction approach. arXiv preprint arXiv:1203.5076.](https://arxiv.org/abs/1203.5076)

* [Mazur and Dale, 2010] [Mazur, P. and Dale, R. (2010). Wikiwars: A new corpus for research on temporal expressions. In Proceedings of the 2010 conference on empirical methods in natural language processing, pages 913–922](https://aclanthology.org/D10-1089.pdf)

* [Zhong et al., 2017] [Zhong, X., Sun, A., and Cambria, E. (2017). Time expression analysis and recognition using syntactic token types and general heuristic rules. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 420–429, Vancouver, Canada. Association for Computational Linguistics.](https://aclanthology.org/P17-1039/)

* [Zarcone et al., 2020] [Zarcone, A., Alam, T., and Kolagar, Z. (2020). Pâté: a corpus of temporal expressions for the in-car voice assistant domain. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 523–530.](https://aclanthology.org/2020.lrec-1.66/)