# Machine Learning Engineer Nanodegree
## Capstone Proposal
Boris Terentiev
October 4th, 2018

## Proposal
### Problem Statement
Movies are typically classified into genres, i.e. drama, comedy, thriller, etc. For marketing purposes studios create a plethora of marketing material such as trailers, posters and merchandise with the intention to give a sense to the viewer what to expect from the movie, especially the genre. Humans are trained to distinguish the little cues in a poster, e.g. smiling people, a dark tone of the poster or the appearance of Jack Black to understand the sentiment and ultimately the genre. I want to use current Machine Learning techniques to see if it is possible to train a machine to make sense of this little cues.

The problem I have chosen is rather synthetic and I can't imagine that this problem occurs in real live, nevertheless the solution might be interesting for real life.

### Datasets and Inputs
[MovieLens](https://grouplens.org/datasets/movielens/) is a non commercial, personalized movie recommendation system and relies on the users to provide metadata for movies. The project provides extensive datasets meant for research or education, such as tags, cast and links to [imdb](https://www.imdb.com/) and [tmdb](https://www.themoviedb.org/?language=de) for a huge set of movies. Additionally one can use the links to tmdb and imdb to retrieve additional information about specific movies, such as numerous posters per movie and missing metadata, e.g. the cast. 

Summarized I am going to use a dataset from MovieLens and extend this dataset by data retrieved from tmdb.

### Solution Statement
The basic idea to tackle this problem is to use multiple (presumably) weak classifiers and create a better performing classifier by combining the weak classifiers. The weak classifiers shall be based on a small piece of movie information e.g. the poster, tags, cast, or the description of the movie (the final implementation may rely only on a subset of these). This way I can use specialized architectures and techniques, e.g. transfer learning for image and text classification for single subtasks. 

More specifically:

The first goal is to examine if a deep neural network (DNN) trained by means of transfer learning is able to get a sense of the cues in a movie posters and to make better predictions than a random model. A possible problem may be that poster styles change over time significantly, e.g. a poster for Ben Hur from 1959 could be a poster for a light-hearted comedy of today. Therefore it may be sensible to restrict the time frame.

Then I will try to create classifier based on movie metadata.

In a final state I am going to combine this classifiers into a single classifier, either by means of Ensemble Learning or by training additional layers in a neural net.

### Benchmark Model
Preliminary analysis of the dataset showed that the movies are assigned to 1.9 genres on average with 18  genres in total. The simplest possible model assigns a single genre randomly to a movie. It is also the best possible model based on random assignment of genres, as it is more probable to make a mistake by assigning a second or more genres given the average count of genres per movie and the total count of genres.

### Evaluation Metrics
The problem of assigning movie genres is a multi label classification problem. One way to evaluate the performance of a model that solves such a problem is the hamming loss. [The hamming loss is the fraction of wrong labels to the total number of labels](https://en.wikipedia.org/wiki/Multi-label_classification) 

### Project Design
First I have to enrich the MovieLens  dataset with the movie posters. I will do this by using the links to tmdb, provided in the MovieLens datasets.

Next step is to analyze and filter the data for movies that can't be processed, e.g. no poster, no genres, etc.

Next we are going to split the data into training, validation and testing sets.

Then we will create a classifier based solely on posters. This classifier will be a deep neural network (DNN) and will be trained by means of transfer learning. The framework of choice will be Keras as it makes the application of transfer learning simple. The training loss will be cross entropy.

I will choose the best model by applying the hamming loss to the validation set and assess the performance of the final model by applying the hamming loss to the test set.

Next step will be to create classifiers that use movie metadata.

Again the performance of the classifiers is assessed by calculating the hamming loss of the test set.

Next I will choose a way (e.g. by adding additional neuron layers) to combine the resulting classifiers into a single and hopefully more accurate classifier.