import json
from components import *

def calculate(original_query,ranker,reader,number):
    ranker = PassageRanker()
    lan = detector.predict(original_query)
    query_generator = QueryGenerator(lan)
    query = query_generator.generate_query(original_query)
    detector = LanguageID()
    reader = DocumentReader(lan)
    extractor = DocumentExtractor(lan)
    
    
    library = extractor.get_documents(query)
    
    
    results = ranker.rank(library,original_query,ranker)
    
    answers = reader.answer(original_query,results,n_results=number)
    
    return answers

def handler(event, context):
    try:
        
        # loads the incoming event into a dictonary
        data = json.loads(event['body'])
        # uses the pipeline to predict the answer
        answer = calculate(data['query'],data['ranker'],data['reader'],int(data['number']))
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type" : "application/json",
            "Access-Control-Allow-Headers" : "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
            "Access-Control-Allow-Methods" : "OPTIONS,POST,GET,ANY,DELETE",
            "Access-Control-Allow-Credentials" : True,
            "Access-Control-Allow-Origin" : "*",
            "X-Requested-With" : "*"

            },
            "body": json.dumps(answer)
        }
    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            "body": json.dumps({"error": repr(e)})
        }