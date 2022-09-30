from flask import Flask, request, Response, jsonify
import pickle
import json
import os
from flask_cors import CORS, cross_origin
from components import *


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'




@app.route('/')
@cross_origin()
def index():
    return 'Hello World'

@app.route('/recieve', methods=['POST'])
@cross_origin()
def recieveAnswer():
    data = request.get_json()
    result = [{
        'answer': 'respuesta',
        'title':'titulo',
        'content':'se extrajo de este texto',
        'url': 'link',
        'score':10
    }]

    result = calculate(data['query'],data['ranker'],data['reader'],int(data['number']))
    return jsonify(result)


def calculate(original_query,ranker,reader,number):
    ranker = PassageRanker()
    query_generator = QueryGenerator(lan)
    query = query_generator.generate_query(original_query)
    detector = LanguageID()
    lan = detector.predict(original_query)
    query_generator = QueryGenerator(lan)

    reader = DocumentReader(lan)
    extractor = DocumentExtractor(lan)
    
    
    library = extractor.get_documents(query)
    
    
    results = ranker.rank(library,original_query,ranker)
    
    answers = reader.answer(original_query,results,n_results=number)
    
    return answers
    


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='8080',)#ssl_context='adhoc')