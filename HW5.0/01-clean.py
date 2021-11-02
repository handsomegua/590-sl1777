# read the data
from keras import preprocessing 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import codecs
import re
from nltk.tokenize import sent_tokenize
import os
import numpy as np
from keras.utils.np_utils import *

corpus_folder = os.getcwd()+'/'+'corpus'
text = []
label = []
corpus_text = os.listdir(corpus_folder)

for text_file in corpus_text:
    with codecs.open(corpus_folder+'/'+text_file, 'r', encoding='utf-8') as fp:
        result = fp.read().strip()
    result_1 = sent_tokenize(result)
    new_result = list()
    for i in result_1:
        i = re.split('\r|\n',i)
        i = [x for x in i if x]
        i = ' '.join(i)
        new_result.append(i)
    # tokenizer.fit_on_texts(new_result)
    # sequences = tokenizer.texts_to_sequences(new_result)
    # pad_data = pad_sequences(sequences,maxlen=max_len)
    for sentence in new_result:
        text.append(sentence)
        label.append(re.split('.txt',text_file)[0])

for index in range(len(label)):
    if label[index] == 'Last_Plunge':
        label[index] = 0 
    elif label[index] == 'Odyssey':
        label[index] = 1 
    elif label[index] == 'Wild_Animals':
        label[index] = 2

text_data = open('data.txt','w')
for sentence in text:
    text_data.write(sentence+'\n')
text_data.close()

text_label = open('label.txt','w')
for num in label:
    text_label.write(str(num)+'\n')
text_label.close()








