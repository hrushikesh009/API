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

model = tc.load_model("Recommender_model")

@app.route('/<string:UserID>', methods = ['GET'])
def method_name(UserID):
    
    Recommended_products = list(model.recommend([UserID])['ProdID'])
    
    return jsonify({'Recommended_Products': Recommended_products}) 

if __name__=='__main__':
    app.run(debug=True)
