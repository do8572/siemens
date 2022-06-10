# David Ocepek's Data Science Project Competition journal

## March 2022 (22h)

* 1. (4h): Meeting. + Initial analysis of dataset. + Discussion/organization with team.
* 2. (8h): Read reserch papers on recommender systems, followed up with tutorials and researched viable python library's. The following algorithms are supported in python MF, FM, FFM, DeepFM, RNN. Non-supported methods include HRNN-meta recommender system, graph NN also appear to be promising though will be challenging to implement.
* 3. (6h): Framework so that we don't have to reimplement common parts of our code multiple times. (Failure due to poor organization - on my part.) 
* 4. (4h): Spotlight implementation of deepFM.

## April 2022 (32h)

* 1. (10h): Analizing FuxiCTR as alternative library for deepFM. (Spotlight lacks support for categorical features.)
* 2. (10h): Analizing & building model using tensorflow deepCTR. (Enables greater control over negative sampling than FuxiCTR.)
* 3. (6h): Fixing Spotlight deepFM. (Due to some yet un-discovered bugs hit rate is 0.0 - TO BE FIXED.)
* 4. (6h): Meetings + miscelenious (journal, report, etc.)

## May 2022 (25h)

* 1. (18h): Implementing deepFM from scratch (implemntation also includes 1st and 2nd order FM - both for comparison with Matej's FM model as well as a check whether our methodologies match), since I wasn't able to find an implementation that meet all of our requirements
* 2. (3h): Preparation of material for Siemens meeting.
* 3. (4h): Meetings (include meeting's with teamates and Siemens)

## June 2022 (24h)

* 1. (6h): Tunning model hyperparamaters, changing deepFM, 1st and 2nd order FM to return probabilities for True/False by adding final softmax layer.
* 2. (2h): Meeting discusion of future work, discrepancy in methodology was found (I was predicting for customer - day pairs, while Matej was predicting for entire period of 4 months.) - FIX NEEDED: We decided Matej solution fit our problem better, so alterations needed.
* 3. (4h): Adding top k hit rate evaluation, making sure CV works.
* 4. (3h): Modifying program to predict for time period of 4 months.
* 5. (9h): Training on all negative samples and unifying methodologies.
* 7. (5h): Add LSTM DeepFM.
* 8. (7h): Add hyperparameter optimization for all models.
* 9. (5h): Graphing, plotting, changing results.
* 10. (8h): Writing preliminray report.

## Total: 128.5h
