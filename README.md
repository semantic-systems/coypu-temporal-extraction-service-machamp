# Introduction

This repository provides the models and the source code for the master's thesis "Extraction and Classification of Time in Unstructured Data" (2023).
Furthermore, it describes the steps required to reproduce the thesis results.
The transformer-based models are finetuned to find and classify temporal expressions in unstructured text.

The models produced by the thesis utilize the two frameworks, UIE and MaChAmp.
Both repositories were forked in August 2023 and modified to suit the problem of temporal extraction and classification.
Most changes were applied to the evaluation and dataset preprocessing scripts.
The scripts for finetuning and inference remain very close to the original versions:

* Unified Structure Generation for Universal Information Extraction (UIE) [Lu et al., 2022] - [GitHub Link](https://github.com/universal-ie/UIE)
    * UIE is a sequence-to-sequence framework that extracts various information extraction targets into a graph structure called "Structured Extraction Language".
    It is based on the T5 library [Raffel et al., 2020].
* Massive Choice, Ample Tasks (MACHAMP) [van der Goot et al., 2020] - [GitHub Link](https://github.com/machamp-nlp/machamp)
    * MaChAmp is a multitask learning framework.
    In this thesis, it is used to train BERT-based models in a single-task fashion. [Lu et al., 2022]



# Finetuned UIE models
| Dataset                        | URL                                                                                                                          |
|-------------------------------:|------------------------------------------------------------------------------------------------------------------------------|
| TempEval-3 (large model)       | [Google Drive Link](https://drive.google.com/file/d/16cZBawptKC6kTv99AvuHFC3wKGkMfcMg/view?usp=sharing)                      |
| WikiWars (large model)         | [Google Drive Link](https://drive.google.com/file/d/1lgDVxx2QfZuLEEx1x2JeCMsC7DboQV00/view?usp=sharing)                      |
| Tweets (large model)           | [Google Drive Link](https://drive.google.com/file/d/1mZvdiq1_nmNv93Bb12xSCScNCYafxIXF/view?usp=sharing)                      |
| Fullpate   (large model)       | [Google Drive Link](https://drive.google.com/file/d/1YkDNhBmcAMxFGaJ7eTUPp3wGjSMHVFVi/view?usp=sharing)                      |



# References
* [Lu et al., 2022] [Lu, Y., Liu, Q., Dai, D., Xiao, X., Lin, H., Han, X., Sun, L., and Wu, H. (2022). Unified structure generation for universal information extraction. arXiv preprint arXiv:2203.12277.](https://aclanthology.org/2022.acl-long.395/)

* [van der Goot et al., 2020] [van der Goot, R., Üstün, A., Ramponi, A., Sharaf, I., and Plank, B. (2020). Massive choice, ample tasks (machamp): A toolkit for multi-task learning in nlp. arXiv preprint arXiv:2005.14672.](https://arxiv.org/abs/2005.14672)

* [Raffel et al., 2020] [Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485– 5551](https://arxiv.org/abs/1910.10683)




[comment]: <> (#Data #Quickstart #Running UIE Models #Running MaChAmp Models #Finetuning Models)
