import nltk
from nltk.translate.bleu_score import SmoothingFunction

def getBlueScore(file_contents):
  cc = SmoothingFunction()
  i = 0
  bleu_4_scores = []
  for i in range(len(file_contents)):
      if len(file_contents[i]) != 2:
        print(file_contents)
        break
      reference_res = file_contents[i][0]
      candidate_res = file_contents[i][1]
      references = reference_res.split(' ')
      hypothesis = candidate_res.split(' ')
      if len(references) >= 4 and len(hypothesis) >= 4:
          BLEUscore = nltk.translate.bleu_score.sentence_bleu([references], hypothesis, smoothing_function = cc.method2)
      elif len(references) >= 3 and len(hypothesis) >= 3:
          BLEUscore = nltk.translate.bleu_score.sentence_bleu([references], hypothesis, weights = (1.0/3, 1.0/3, 1.0/3), smoothing_function = cc.method2)
      elif len(references) >= 2 and len(hypothesis) >= 2:
          BLEUscore = nltk.translate.bleu_score.sentence_bleu([references], hypothesis, weights = (0.5, 0.5), smoothing_function = cc.method2)
      else:
          BLEUscore = nltk.translate.bleu_score.sentence_bleu([references], hypothesis, weights = [1], smoothing_function = cc.method2)
      bleu_4_scores.append(BLEUscore)
  return sum(bleu_4_scores)/float(len(bleu_4_scores))
