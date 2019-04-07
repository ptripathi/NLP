# About
English - Hindi Neural Machine Translation model experiments using new tensorflow 2.0 (alpha) keras apis.

# Orginazation:
- models
-- utils.py - shared utility functions  

-- score.py - utility module to compute BLEU-4 score on the translation results  

-- data_prep.py - utility functions to help data preprocessing  
- model_n folders
-- model artifacts such as encode, decoder  

-- train: notebook to train the model  

-- test: notbook to test the model  


# Process

![nmt process](nmt_process.png)



# References
* Anoop Kunchukuttan, Pratik Mehta, Pushpak Bhattacharyya. The IIT Bombay English-Hindi Parallel Corpus. Language Resources and Evaluation Conference. 2018.
* Tensorflow Tutorials: https://www.tensorflow.org/tutorials . Some code borrowed from tutorials for fast prototyping.
