# Influence-Relationship-Classification
## About The Project
The project aims to classify influence relationships in Online Health Communities. Will add the published paper link later.
## Input Datasets
### [breast_cancer.csv](./intermediate_data)
The file needs to be unziped before using. It is the Online Health Community (OHC) data for training embeddings. We crawled the data from [Cancer Survivors Network's Breast Cancer Forum](https://csn.cancer.org/forum/127).
There are 3 colums:
1. thread_id: The id of the input thread.
2. init_post: The raw text of the initial post of the thread.
3. replies: The concatenation of all raw text of the reply posts.
### [influence_data.csv](./influence_data/influence_data.csv)
This is the input dataset for influence classification. There are 7 columns:
1. thread_id: The id of the input thread.
2. initial_post: The raw text of the initial post of the thread.
3. reply_post: The raw text of other OHC users' reply to the initial post.
4. initial_author_reply: Initial post's author's reply to reply_post.
5. label: The label presents whether this is an influence relationship.
6. AB: The label presents whether initial_post and reply_post are relevant.
7. BC: The label presents whether reply_post and initial_author_reply are relevant.
### [pairs.csv](./relevant_classification_model/pairs.csv)
This is the input dataset for relevance classification. There are 4 columns:
1. post1: The embedding vector of the first post in the relevance pair. ([bert](https://pypi.org/project/bert-embedding/), [word2vec](./relevant_classification_model/post1_embedding.csv.zip))
2. post2: The embedding vector of the second post in the relevance pair. ([bert](https://pypi.org/project/bert-embedding/), [word2vec](./relevant_classification_model/post2_embedding.csv.zip))
3. label: The label presents whether post1 and post2 are relevant.
4. train_or_test: The label presents whether the pair are used for training or testing.
## Built With
Following are the major frameworks/libraries used to bootstrap this project:
* [pandas](https://pandas.pydata.org/)
* [sklearn](https://scikit-learn.org/stable/)
* [keras](https://keras.io/)
* [numpy](https://numpy.org/)
* [nltk](https://www.nltk.org/)
* [matplotlib](https://matplotlib.org/)
* [bs4](https://pypi.org/project/bs4/)
* [gensim](https://radimrehurek.com/gensim/)
* [spyder-kernels](https://pypi.org/project/spyder-kernels/)

It is recommended to install all these libs using [conda](https://docs.conda.io/en/latest/index.html#).
## How To Run
1. Change the [model options](./code/influence_calssification.py#L16-L35) based on which model to run.
2. Change the [train or load choices](./code/influence_calssification.py#L153-L156) based on whehter to train new relevance/influence models.
3. Optional: Tune the [parameters](./code/influence_calssification.py#L51-L73) if necessary.
4. Execute [influence_calssification.py](./code/influence_calssification.py) in your IDE or simply run the command: `cd code && python3 influence_calssification.py`.
