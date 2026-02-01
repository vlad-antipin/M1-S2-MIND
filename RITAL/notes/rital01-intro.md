
NLP first, then IR

Textual data is pretty hard to put in a table to furnish for a ML model

It's hard to adapt ML and ANN to NLP, and much worse for IR

RAG - mix of NLP and IR

Evaluation is very tricky, hard to design labels, especially for text generation or for recommendation, IR, dialog systems - a lot of subjectivity and ambiguity

It's very important to know hidden biases in language models  you use


# NLP

#lookup Manning resources

## Applications

Access knowledge, communication, linguistics / cognitive science

To handle the ambiguity issue we can integrate knowledge bases  done by experts

Need to consider variability of meanings

Word's frequency ~ Zipf law  #lookup connection to power law?
$f_{w}(k) \propto \frac{1}{k^\theta}$ for kth most frequent term

first words are "empty" like articles the, a .... we often get rid of them

# Tasks

Tasks: different levels
- document level
	- indexing, classification
	- counting / survey
	- clustering (topic analysis)
	- segmentation
	- automated summary NB: extractive summary limits hallucinations
- sentence / paragraph
	- question answering (QA) - with or without the text of reference
	- coreference resolution - e.g. which persons are referenced by pronouns
	- information extraction
		- good for knowledge base construction, NER?
		- for automatic contract reader
		- in medicine ...
	- dialog 
		- chatbot
	- disambiguation 
	- part-of-speech tagging (POS)
		- can reconstruct grammatical tree
	- sentiment analysis
	- translation
	- ....
- word
	- often not very useful alone
	- spelling connection
	- semantic distances
- stream (multimodal: text-time)
	- topic appearing, vanishing 
	- funny application: detection of influenza epidemics

Profiling of users ...

Easy tasks: spam detection, POS, NER
Very hard: QA, paraphrase, summarizations, dialog

# Methods and evaluation

Models:
- state machines and pattern detection there were first models
- for simple tasks may prefer simple models to not waste resources
- same methodology as in DALAS, just issues to encode the data - put text in a vector

Standard pipeline:
- preprocessing - encoding, punctuation...
- formatting - text into a vector
- learning #lookup  CRF in NLP
- hyperparameter optimization

Important to acknowledge where your model doesn't work

Bad to be limited by basic evaluation like accuracy, precision, recall... (because subjectivity and ambiguity of outputs)

# Task examples

Project will be on: (5 weeks for NLP)
Document Classification task - sentiment, speaker classif. ...
Review sentiment analysis

Problems:
- big corpus - huge vocabulary 
	- log. reg. , SVM, naive bayes...
- hard to model  sentence structure
	- can remove the structure... - bag of words (BoW) model (frequencies of words)
- word polymorphism
- synonyms
- problem of high dimensionality
	- remove useless words

Naive Bayes - hypothesis of word independence conditional to class, can consider then product of variables

Classical problems: regularization, class balancing

Big vocabulary - very sparse frequency vectors

Do a evaluation protocol before touching the model

#todo resources for project in the end of CM1