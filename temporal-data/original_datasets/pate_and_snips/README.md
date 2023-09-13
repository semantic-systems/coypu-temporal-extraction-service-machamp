## PÂTÉ: A Corpus of Temporal Expressions for the In-car Voice Assistant Domain

We present PÂTÉ: A Corpus of Temporal Expressions for the In-car Voice Assistant
Domain. PÂTÉ is a dataset of natural-language commands containing
temporal expressions for an AI voice assistant.

The dataset is presented in:

Zarcone, Alessandra and Alam, Touhidul and Kolagar, Zahra (2020). "PÂTÉ: A
Corpus of Temporal Expressions for the In-car Voice Assistant Domain".
Proceedings of the 12th Conference on Language Resources and Evaluation (LREC).
Marseille, France, May 11-16, 2020.

The dataset is annotated according to the TimeML/TIMEX3 annotation guidelines
(Saurı et al., 2006; TimeML Working Group, 2009). In our paper we also describe
how we annotated cases which were not covered by the guidelines.

We also provide an annotation of the Snips dataset (Coucke et al., 2018)
following the same guidelines.

For both datasets, we used "**2019-11-18**" as the reference date **t0** and,
in case a reference time was also required (e.g., *five minutes from now*), we
used "**2019-11-18T08:00**" as reference time **t0**.

### PÂTÉ

The file `pate.json` contains the PÂTÉ dataset. The dataset consists of 480
single commands directed to a virtual assistant, out of which 353 sentences 
contain time expressions. Each command is annotated with the corresponding
intent, a domain or scenario, and *datetime* entities. No other entities are
annotated.

Each command is represented as a JSON object with the following fields:

- **text**: the command given to the virtual assistant
- **intents**: an annotation of the user intent, *Event.book*, *Event.new*,
*Event.search*, *Event.change*, or *Event.delete*
- **domain**: an annotation of the domain / scenario for the command (e.g. *hotel*
	for hotel bookings, *hairdresser* for hair salon bookings, etc.)
- **entities**: entities mentioned in the command. We only provide annotation of
*datetime* entities, that is of time expressions mentioned in the command
	**entities** contain the following fields:

	- **values**: the time expression
	- **entity**: the entity type (which in this dataset is always *datetime*)
	- **TIMEX3**: a list of TIMEX3 expressions in the command, each of them
	containing one or more of the following attributes:

		- **expression**: the time expression (which may be a substring of **value**)
		- **tid**: the unique id number of the time expression in the dataset
		- **type**: the TIMEX3 type of the expression (DATE, TIME, DURATION or SET)
		- **value**: the normalized value of the time expression
		- **mod** (optional): quantification modifier of the time expression
		- **anchorTimeID** (optional): the **tid** which the time expression has been anchored to
		- **beginPoint** (optional): the **tid** of the starting point of the time expression
		- **endPoint** (optional): the **tid** of the ending point of the time expression
		- **uncertain** (optional): Boolean, True when a time expression is
		labeled as uncertain

### Snips

The Snips dataset is annotated following the same guidelines as the PÂTÉ dataset.

The dataset contains all Snips commands (Coucke et al., 2018) for the
*BookRestaurant* intent, and  consists of two separate train and validation files
(respectively, `snips_train.json` and `snips_valid.json`), containing 708
sentences in total, out of which 697 sentences contain time expressions.

Each command is represented as a JSON object consisting of a list. Each list
contains:

- **text**: a chunk of the command, identifying either the span of an entity or
a span without entities
- **entity**: the type of the entity, if the **text** chunk identifies an entity
- **TIMEX3**: a list of TIMEX3 expressions in the chunk, each of them
containing one or more of the following attributes:

    - **expression**: the time expression (which may be a substring of **text**)
		- **tid**: the unique id number of the time expression in the dataset
		- **type**: the TIMEX3 type of the expression (DATE, TIME, DURATION or SET)
		- **value**: the normalized value of the time expression
		- **mod** (optional): quantification modifier of the time expression
		- **anchorTimeID** (optional): the **tid** which the time expression has been anchored to
		- **beginPoint** (optional): the **tid** of the starting point of the time expression
		- **endPoint** (optional): the **tid** of the ending point of the time expression
		- **uncertain** (optional): Boolean, True when a time expression is
		labeled as uncertain
