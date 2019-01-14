# Movie Genres ML

### Prerequisites

python 3.6

GPU installation that works with tensorflow-gpu

TMDB API Key

### Installation

Install the necessary libraries with

```python
pip install -r requirements.txt
```



### Resources

#### Code

The code is structured in three notebooks that should be performed in the following order

<div style="text-align:center; font-weight:bold">
    DataRetrieval.ipynb  > Preprocessing.ipynb > Training.ipynb
</div>

The intermediate results are cached in the *data* folder and are provided together with the source code. That is each jupyter notebook should be runnable on its own.

- **DataRetrieval.ipynb** is responsible to collect data from MovieLens project and TMDB. **You have to add the tmdb API key to be able to run this notebook.**

- **Preprocessing.ipynb** is about preprocessing the data before training the models.

- **Training.ipynb** implements the 5 models and the training procedure

#### Data

The data that is retrieved and preprocessed in **DataRetrieval.ipynb ** and  **Preprocessing.ipynb** is cached in the *data* folder.

- **ml-20m** is the MovieLens data
- **posters** are the movie posters downloaded from tmdb
- **extended_movie_posters_data.csv, extended_movie_data_with_local_files.csv, movie_training_data.csv** are different stages of preprocessing
- **classification_examples** are examples for classification with the best model.



