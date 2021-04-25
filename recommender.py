from flask import Flask, jsonify, request, make_response
import turicreate as tc
import numpy as np
import string
import random
import re
import gc
import time
import json

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def method_name(Uid):
    dataset = request.get_json() #dataset received in dictionary format
    given_sub = dataset['Sub_Category']
    with open('test.json', 'w') as f:
        json.dump(dataset, f)
    
    fashion_dataset = tc.SFrame.read_json('test.json',orient = 'lines')
   
    fashion_dataset['brand'] = fashion_dataset['brand'].apply(Filtering_brand)
    Title_without_punctuation = fashion_dataset['title'].apply(remove_punctuation)
    Description_without_punctuation = fashion_dataset['description'].apply(remove_punctuation)
    fashion_dataset['title_word_count'] = tc.text_analytics.count_words(Title_without_punctuation)
    fashion_dataset['title_tf_idf'] = tc.text_analytics.tf_idf(Title_without_punctuation)
    fashion_dataset['description_word_count'] = tc.text_analytics.count_words(Description_without_punctuation)
    fashion_dataset['description_tf_idf'] = tc.text_analytics.tf_idf(Description_without_punctuation)
    fashion_dataset['price'] = fashion_dataset['price'].apply(normalization)
    # fashion_dataset['Rank'] = fashion_dataset['Rank'].apply(normalization)

    for x in features:
        if x in given_sub:
            fashion_dataset[x] = 1
        else:
            fashion_dataset[x] = 0

    model = tc.load_model("final_linear.model")

    predicted_price_log = model.predict(fashion_dataset)

    predicted_price = np.expm1(predicted_price_log)

    return jsonify({'price':predicted_price[0]}) #can jsonify dictionary and feed it to the output, maybe wrong syntax

if __name__=='__main__':
    app.run(debug=True)
