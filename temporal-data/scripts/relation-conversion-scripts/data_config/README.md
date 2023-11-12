The TempEval-3 datasets consist of three types of information:
* Temporal entities in TIMEX3 format (date, time, duration, set)
* Temporal events, which may also be considered as entities in terms of NER (named entity recognition).
* Temporal relations between events pairs (EE), temporal entity pairs (TT) and pairs of entities with events (TE, ET)

The data-config files are required for the UIE converter.
The files tell it information about the dataset.
The "relation" data config files are in the "tempeval_relation" directory.
The "tempeval_entity" directory creates regular temporal extraction data config files i.e. they only contain temporal entities.
The "tempeval_event" directory creates data config files that contain temporal entities and events.
The idea of having this distinction is to test variable amounts of information given to a transformer-based framework like UIE.