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

def Filtering_brand(s):
    if s.lower() == '' or s.lower() == 'unknown':
        return 'Unknown'
    return s

def remove_punctuation(text):
    translator = text.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    return text

def normalization(n):
    return np.log1p(n)

features = ['ATV',
 'Accents',
 'Accessories',
 'Accessory',
 'Active',
 'Alternative',
 'Amazon',
 'Apparel',
 'Aprons',
 'Arts',
 'Athletic',
 'Automotive',
 'Baby',
 'Backpacks',
 'Bags',
 'Bands',
 'Bar',
 'Bath',
 'Bathroom',
 'Beading',
 'Beauty',
 'Boas',
 'Bottle',
 'Boxes',
 'Boys',
 'Bracelets',
 'Brooches',
 'Card',
 'Care',
 'Cases',
 'Cats',
 'Cell',
 'Charm',
 'Charms',
 'Chefs',
 'Climbing',
 'Clothing',
 'Coats',
 'Containers',
 'Cooling',
 'Corkscrews',
 'Corsets',
 'Cosplay',
 'Costumes',
 'Craft',
 'Crafts',
 'Crossbody',
 'Cultural',
 'Curtain',
 'Curtains',
 'Dcor',
 'Decorative',
 'Dining',
 'Dogs',
 'Down',
 'Dress',
 'Dressing',
 'Earrings',
 'Eastern',
 'Electronics',
 'Fan',
 'Fans',
 'Feather',
 'Feathers',
 'Food',
 'Footwear',
 'Furniture',
 'Gadgets',
 'Games',
 'Garden',
 'Gear',
 'Girls',
 'Glasses',
 'Grocery',
 'Handbags',
 'Hats',
 'Health',
 'Heating',
 'Helmet',
 'Home',
 'Hooks',
 'Household',
 'Improvement',
 'Industrial',
 'Instruments',
 'Jackets',
 'Jewelry',
 'Keychains',
 'Keyrings',
 'Kids',
 'Kitchen',
 'Launchpad',
 'Lawn',
 'Linens',
 'Liners',
 'Lingerie',
 'Lounge',
 'Luggage',
 'Lunch',
 'Making',
 'Memorials',
 'Men',
 'Mens',
 'Middle',
 'Motorcycle',
 'Musical',
 'Necklaces',
 'Novelty',
 'Office',
 'Openers',
 'Organization',
 'Organizers',
 'Ornaments',
 'Outdoor',
 'Outdoors',
 'Pads',
 'Patio',
 'Pendants',
 'Performance',
 'Personal',
 'Pet',
 'Phones',
 'Pins',
 'Play',
 'Posters',
 'PreKindergarten',
 'Preschool',
 'Pretend',
 'Prints',
 'Products',
 'Props',
 'Protective',
 'Quality',
 'Rain',
 'Recreation',
 'Reusable',
 'Robes',
 'Role',
 'Room',
 'Scientific',
 'Sets',
 'Sewing',
 'Shoe',
 'Shoes',
 'Shop',
 'Shops',
 'Shower',
 'Sleep',
 'Sports',
 'Storage',
 'Stud',
 'Supplies',
 'TShirts',
 'Table',
 'Tabletop',
 'Tees',
 'ToGo',
 'Toddler',
 'Tools',
 'TopHandle',
 'Tops',
 'Totes',
 'Towels',
 'Toys',
 'Traditional',
 'Travel',
 'Trench',
 'Up',
 'Urns',
 'Utensils',
 'Vests',
 'Wall',
 'Wallets',
 'Watch',
 'Watches',
 'Wear',
 'Wine',
 'Women'
 ]

@app.route('/', methods = ['GET', 'POST'])
def method_name():
    dataset = request.get_json() #dataset received in dictionary format
    given_sub = dataset['Sub_Category']
    with open('test.json', 'w') as f:
        json.dump(dataset, f)
    
    fashion_dataset = tc.SFrame.read_json('test.json',orient = 'lines')
    # def Filtering_rank(s):
    #     if s == []:
    #         return 3188137
    #     return int(s[0])
    # fashion_dataset['Rank'] = fashion_dataset['Rank'].apply(Filtering_rank)
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
