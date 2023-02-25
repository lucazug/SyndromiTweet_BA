#### README-file for Bachelor's thesis: Luca Zug (Matrikelnummer: 03729671)

# Files and Folders

## Python Files

#### main /
  - funcs.py (Relevant background functions)
  - keyword_selec.py (Preprocessing list of symptoms into search query and pulling tweets through the Twitter API)
  - keyword_selec_model.py (Analysing manually labelled tweets and return significant keywords and "foundwords")
  - pull_tweets.py (Main API call to pull main Twitter data)
  - to_manual_coding.py (Randomly select main Twitter data for labelling.)
  - nlp_train_bert.py (Train the model on hand labelled data, save model.)
  - nlp_prediction_bert.py (Predict main Twitter data from pretrained model.)
  - evaluation.py (Evaluate predictions for tweets and correlate with official data sources.)

## Data in GitHub Repo

#### main /
  - Symptome.csv (List of symptoms used in keyword_selec.py)
  - Influenza_RKI.csv (RKI data on infections used in evaluation.py)
  - Significant_Keywords.csv (List of keywords for Twitter search query out of keyword_selec_model.py into pull_tweets.py)
  - Significant_Foundwords.csv (List of keywords for Twitter search query out of keyword_selec_model.py into pull_tweets.py)

#### main / array_data /
- y_train_array.npy
- y_valid_array.npy
- y_test_array.npy

## Data in Cloud Storage

### Data_SyndromiTweet /
 - inputs_train_embeddings.pbz2 (Word embeddings for training data)
 - inputs_valid_embeddings.pbz2 (Word embeddings for validation data)
 - inputs_test_embeddings.pbz2   (Word embeddings for test data)
 - lostpredict.json (JSON-file with list of tweets that caused errors during embeddings retrieval)
 - predicted_tweets_MAIN.csv (Table of tweet predictions returned by the model)

#### Data_SyndromiTweet / _maincall_embeddings
- n = 57 mini-batches of compressed pickle files containing word embeddings, {n}_embeddings_main_call.pbz2

#### Data_SyndromiTweet / twitter_data /
- Twitter_main_final_fixed.csv (Main dataset)
- 230103_Twitter_manual_coding_finished.csv (Model training input data)
- Twitter_Keyword_Selec_final.csv (Keyword selection input data)

## Necessary Packages and Dependencies

* Please install the necessary packages by running
  "pip install -r requirements.txt" in the Terminal

pandas, numpy, time, requests, matplotlib.pyplot, os, re, shutil, string,
tensorflow, tensorflow.keras, keras.utils.vis_utils, collections,
sklearn.preprocessing, sklearn.model_selection, sklearn.metrics, sklearn.utils,
sklearn.feature_extraction.text, sklearn.feature_selection, scipy.stats
sklearn.feature_selection, tensorflow.keras.preprocessing.sequence,
kernelapp (from ipykernel), json, pydot, pickle, datetime, random, math, itertools,
requests

* The word embeddings that are fed into the CNN were obtained through the
  Huggingface Inference API with the G-BERT base model developed by deepsetAI.
  The API tokens are not included in the above files. The actual word embeddings
  are save in pickle files for the training, validation and test set, respectively.

## Notes

- The code is commented and formatted according to Rossum, Warsaw and Coghlan's "Style Guide for Python Code" (https://peps.python.org/pep-0008/#code-lay-out)
- Please note that some portion of some code (.py files) was used for API calls
  that were only necessary once because the retrieved data was saved. The API
  calls were left in the code for grading purposes, but they were commented out.

## MIT License

Copyright (c) [2023] [Luca Zug]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING ,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.