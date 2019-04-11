import os, re
import pandas as pd
import numpy as np
import fastText


def preprocess_texts(raw_texts, replacements):
    """
    texts: np.Series containing strings to be preprocessed
    replacements: pairs of what convert to what
    return np.Series with corrected texts
    """
    resulttext = raw_texts.str.lower()
    for [co, naco, _] in replacements.values:
        resulttext = resulttext.str.replace(re.compile(str(co)), str(naco))
    return resulttext


# interesting_labels=pd.read_csv(os.path.join(datapath,'interesting_labels.csv'))

def make_formatted_predictions(clf, texts):
    predictions = []
    for t in texts:
        labels, probs = clf.predict(t, k=1, threshold=0.1)
        result_line = ''
        # print(f'labels:{labels}')
        # print(f'probs:{probs}')
        for i, l in enumerate(labels):
            # print(f'i:{i} l:{l} probs[i]:{probs[i]}')
            l = l.replace('__label__', '')
            result_line += l + ': ' + str(probs[i]) + '; '
        # print(f'result_line: {result_line}')
        predictions.append(result_line)
    return predictions


datapath = '.'
print('loading model...')
classifier = fastText.load_model(os.path.join(datapath, 'model.bin'))
print('loading preprocessing configuration...')
podmiany = pd.read_excel(os.path.join(datapath, 'preproc_dict.xlsx'))

print('loading cases for prediction...')
texts_test = pd.read_csv(os.path.join(datapath, 'cases2predict.txt'), header=None)
print('preprocessing...')
test_set = preprocess_texts(texts_test[0], podmiany)
print('predicting labels...')
result = make_formatted_predictions(classifier, test_set.values)
print('saving predictions...')
pd.Series(result).to_csv(os.path.join(datapath, 'predictions.csv'), index=False)
print('Done! Please find results in predictions.csv')
