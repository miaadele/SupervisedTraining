**GOAL**: train a supervised classifier that learns to recognize and distinguish Early Modern texts that engage in merchant-related discourse versus those that do not
***
In the **BERT** folder:
1. Use baseline BERT model to practice tokenization, work with PCA visualization, and understand masked language modeling.
2. Fine-tune the base model with the Geneva Bible.
3. Cluster documents to disscover the types of texts an unknown Early Modern archive contains.
***
In the **LogisticRegression** folder:
1. script for baseline model, logistic regression  with L2 regularization (ridge-style shrinkage)
2. script for new model, logistic regression with L1 regularization (lasso-style shrinkage)

Print the:
- confusion matrix
- classification report (precision, recall, and F1)
- ROC AUC
- model sparsity diagnostic number
- top 15 positive-weighted words (most predictive of CORE = 1)
- top 15 negative-weighted words (most predictive of NEG = 0)
***
In the **NER** folder:

Use the classifier from the **train_supervised_classifier_week8** folder to do Named Entity Recognition (NER) using spaCy.

Each part of the process is separated into self-contained units of code, labeled accordingly:

**1_load_classifiers.py**:
- load the TF-IDF vectorizer and classifier from the **train_supervised_classifier_week8** folder
- save the classified corpus into "classified_texts.json"


**2_chunking.py**:
- pass raw text into a processing pipeline that 1) tokenizes the text, 2) splits sentences, 3) assigns part-of-speech tags, 4) computes syntactic depndencies, and 5) identifies named entitites
- split the texts into fixed-size chunks of 50,000 characters


**3_normalization**:
- remove unnecessary whitespace and trivial formatting so that they do not create artificial duplicates
- lowercase since case use in Early Modern English is not standardized

**4_finetuning**:
- only focus on two labels, GPE and LOC:
- use the dropout regularisation technique to prevent the model from becoming too dependent on any individual neuron
- use the spaCy optimizer called Adam, configured with a smaller learning rate appropriate for fine-tuning
- set epoch to 20; the model goes through the full set of examples 20 times. Each pass gives hte model another chance to refine its weights. Before each epoch, the training examples are put in a new random order. This matters because the model updates its weights after every single example, and those updates accumulate.

  
**5_rerun_NER_with_finetuned_model**:
- take a different chunking approach: back up to the nearest whitespace so as to not split tokens
- the tuned model is labeling fewer spans as GPE/LOC overall

**tuned_counts.py**:
- take a closer look at the changes between the base counts and fine-tuned counts

***
Results: 
- the baseline model overproduced certain contemporary or highly frequent place names and under recognized Early Modern trade locations.
- the fine-tuned model reduces some modern bias while increasing recognition of historically salient geographic terms and multi-word colonial phrases
- there are small artifacts of overfitting
***
In the **train_supervised_classifier_week8** folder:
Use Word2Vec to
1. Define the concept "merchant" to use as the "seed" concept
2. Segment texts into chunks
3. Create a weakly supervised training dataset using a seed term and related terms

   
   This is accomplished with a tiered search-word list, weak labels, CORE v. NEG dataset.

4. Train a baseline text classifier using TF-IDF features and logistic regression
***
