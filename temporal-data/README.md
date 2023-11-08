# Datasets Overview

The structure of the dataset folder looks like this:

```text
temporal-data
├── entity                  # Contains all the converted datasets used in the thesis
├── original_datasets       # Contains the original versions of the datasets
├── relation                # Contains the temporal relation extraction datasets in UIE-format
```

For finetuning and inference, either the original datasets can be converted into the required format (using the provided scripts) or the already converted datasets may be used.
The main folder with all the different dataset variations is the "entity" folder.
It contains the single- and multiclass datasets in three formats: bio, jsonlines and uie.

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

The UIE models use a JSON format with four temporal classes: date, time, duration, set.
Similar to T5 (text-to-text) models, each data-entry consists of a prompt called "SSI" and target graphlike structure called "SEL". 

# Original datasets sources

* [Fullpate](https://zenodo.org/records/3697930#.ZBwzbi00hQI): consists of Pate and Snips datasets.
* [WikiWars (tagged)](https://github.com/satya77/Transformer_Temporal_Tagger): can be found under data


# UIE Data-Structure

# MaChAmp Data-Structure

# Generic JSON Data-Structure

# Conversion scripts

# References
* [UzZaman et al., 2013] [UzZaman, N., Llorens, H., Derczynski, L., Allen, J., Verhagen, M., and Pustejovsky, J. (2013). Semeval-2013 task 1: Tempeval-3: Evaluating time expressions, events, and temporal relations. In Second Joint Conference on Lexical and Computational Semantics (* SEM), Volume 2: Proceedings of the Seventh International Workshop on Semantic Evaluation (SemEval 2013), pages 1–9. ](https://aclanthology.org/S13-2001.pdf)