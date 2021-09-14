This repository contains the code and report of my NLP course project (part of my MSc. in Social Data Science at the University of Oxford). I jointly predict the helpfulness and 5-star rating of Amazon reviews using various NLP models and combinations of features.

The full reports is available in the "Full report.pdf" file. Most of the code as well as a detailed breakdown of my methodology is available in the "Main_Code.ipynb" file. I also used Google Colab for their GPUs to train my LSTM models, which is available in the "LSTM_Code_Colab.ipynb" notebook. Below I include various key figures and a short summary of my work

## Exploratory Analysis
After extracting the data from [https://jmcauley.ucsd.edu/data/amazon/](https://jmcauley.ucsd.edu/data/amazon/), cleaning it (see my full report for details on the cleaning process) and pickling it for faster loading, I used various techniques to explore the data. This included Jaccard similarities, rank frequency plots of the vocabularies, Vader sentiment as well as Flesch–Kincaid readability scores of each review, mutual information, and t-SNE plots. Below I show the t-SNE plots, and the other analysis' can be found in the "Main_Code.ipynb" notebook.

The t-SNE plots were used to determine the seperability between the classes and utilize the wiki-news-300d FastText embeddings of the content of all reviews (for helpfulness and sentiment seperately for easier understanding). From both Figures it is clear that the classes are not easily seperable in 2D space, which could indicate that the embeddings do not seperate the data very well, or that predicting review helpfulness and valence are challenging clasisfication tasks. I later show that my LSTM models achieve high-accuracy using these word embeddings, so fine-tuning the embeddings was not necessary.
![alt text](https://github.com/plizeeee/NLP-Amazon-Reviews-Project/blob/main/Images/TSNE%20sentiment.PNG)
![alt text](https://github.com/plizeeee/NLP-Amazon-Reviews-Project/blob/main/Images/TSNE%20helpfulness.PNG)

## Models and Performance
I trained 5 models and measured their accuracy using various combinations of features, including the review content (commonly referred to as the Amazon review itself), the review summary (the review title), and the review date. The ensemble LSTM models performed best, and it appears that including additional features provided modest boosts in accuracy compared to using content or non-content alone.
![alt text](https://github.com/plizeeee/NLP-Amazon-Reviews-Project/blob/main/Images/Accuracies%20Movies%20and%20TV.PNG)


I wanted to explore if the additional features were more useful for predicting review helpfulness or sentiment, and from the table below it is clear that relative to a model that uses content alone, models that use other features increase model performance in terms of sentiment rather than helpfulness.
![alt text](https://github.com/plizeeee/NLP-Amazon-Reviews-Project/blob/main/Images/Accuracy%20Breakdown%20Movies%20and%20TV.PNG)

## Analyzing Classifier Errors
Next, I wanted to analyze the types of errors of my classifiers (in addition to the error made at an aggregate level from the confusion matrix, which are available in my main report), as well as which errors were corrected when I included additional features to the classifier (compared to a classifier that used content-features alone). The examples below are from the Naive Bayes classifier, but could be done for all other classifiers explored in my work. For the purposes of not making my report too lengthy, I did not perform this analysis for the other classifiers, although my code could be easily modified for this additional error analysis.
&nbsp;

### Example where the classifier missclasiffied the sentiment of the review.

**Review Summary:** “There's a Fine Line Between Madness and Genius”

**Review Content:** “Right out of the gate this book creeped me out. Being a brunette myself, the title alone was enough to scare me! This was a seriously good book.Words like unhinged, grotesque, shocking, intriguing, and macabre were spinning through my head as I read the stunning and suspenseful pages. In this book, the lines between profiler and killer severely blur constantly. Cristyn West keeps you guessing, "Just WHO is the killer?" And just when you think you've figured it out, she takes you through to another labyrinthine twist.I found myself wondering, "Just who is watching you?" This was a creepy, gritty, horrifying and shocking ride. Right until the last page I found myself exclaiming, "No way!" This book will make you jumpy, especially if you're a brunette. You'll find yourself checking your doors and windows, and looking over your shoulder. You'll catch yourself watching everyone around you, wondering......The suspense is drawn out to the last drop, right up until you discover the shocking truth! To catch a killer, just how far would you go? Read Plain Jane: Brunettes Beware and you will discover that there's a fine line between madness and genius!”

**Classifier predicted class:** Negative

**Actual class:** Positve
&nbsp;

&nbsp;

### Example where the classifier missclasiffied the helpfulness of the review.

**Review Summary:** Shocking Truth

**Review Content:** This is a really good book. I flipped through the pictures and data mostly, but you couldn't help reading the text as well for the information is shocking. There are sharp comparisons between the past and now, in stunning photographic form: Alaska, glaciers around the world, coral bleaching, hurricanes, ocean currents, coastlines, lives affected by the damages and humans' continued invasion. Al Gore's conviction and unique access to information gave this book immense value

**Classifier predicted class:** Helpful

**Actual class:** Unhelpful	
&nbsp;

&nbsp;

### Example where the sentiment was corrected by including the review summary. Note this was only done for the review sentiment, since the classifiers accuracies did not improve when including the review summary in terms of predicting review helpfulness

**Review Summary:** "Eh kinda sucks now”

**Review Content:** “This book was the first finance book I ever read back in the 90s until now. I loved it at the time. I recently bought a copy and it just does not really grab me. I would suggest something by Swedroe or Bogle”

**Classifier predicted sentiment from content alone:** neutral

**Actual AND predicted sentiment using both content and the title:** negative



