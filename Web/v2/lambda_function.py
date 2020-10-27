import json
import os

import sys
sys.path.append("/mnt/access/lib/python/")

from joblib import load
import pandas as pd

def lambda_handler(event, context):
    expected_columns = ['children', 'bmi', 'waist_to_hip_ratio', 'bc_receptor', 'types_of_surgery', 'num_lymph_nodes_removed', 'hormonal_therapy', 'hypertension_and_diabetes', 'heaviness_or_tightness', 'extra_supplementation', 'difficulty_finding_shirt_that_fits', 'dash_score']
    if set(expected_columns) != set([key for key in event]):
        return json.dumps({
            "status": 500,
            "message": "Invalid request body structure. Either some of the expected keys are missing or invalid keys have been put in. Expected keys: {'children', 'bmi', 'waist_to_hip_ratio' ,'bc_receptor', 'types_of_surgery', 'num_lymph_nodes_removed', 'hormonal_therapy', 'hypertension_and_diabetes', 'heaviness_or_tightness', 'difficulty_finding_shirt_that_fits', 'extra_supplementation', 'dash_score'}"
        })
    
    clf = load("/mnt/access/obj/classifier.mdl")
    prediction = clf.predict_proba(pd.DataFrame(
        columns=[
            'Hormonal therapy',
            'Hypertension & diabetes',
            'SS1: Heaviness/tightness',
            'SS2: Hardness/ difficulty finding shirts that fits',
            'Extra supplementation',
            'DASH score',
            'Children',
            'BMI',
            'Waist to hip datio',
            'BC receptor',
            'Types of surgery',
            'Number of lymph nodes removed'
        ],
        data=[[
            event['children'],
            event['bmi'], 
            event['waist_to_hip_ratio'], 
            event['bc_receptor'], 
            event['types_of_surgery'], 
            event['num_lymph_nodes_removed'], 
            event['hormonal_therapy'],
            event['hypertension_and_diabetes'],
            event['heaviness_or_tightness'],
            event['difficulty_finding_shirt_that_fits'],
            event['extra_supplementation'],
            event['dash_score']
        ]]
    ))[0, 0]
    return json.dumps({
        "status": 200,
        "prediction":prediction
    })import json
import os

import sys
sys.path.append("/mnt/access/lib/python/")

from joblib import load
import pandas as pd

def lambda_handler(event, context):
    expected_columns = ['children', 'bmi', 'waist_to_hip_ratio', 'bc_receptor', 'types_of_surgery', 'num_lymph_nodes_removed', 'hormonal_therapy', 'hypertension_and_diabetes', 'heaviness_or_tightness', 'extra_supplementation', 'difficulty_finding_shirt_that_fits', 'dash_score']
    if set(expected_columns) != set([key for key in event]):
        return json.dumps({
            "status": 500,
            "message": "Invalid request body structure. Either some of the expected keys are missing or invalid keys have been put in. Expected keys: {'children', 'bmi', 'waist_to_hip_ratio' ,'bc_receptor', 'types_of_surgery', 'num_lymph_nodes_removed', 'hormonal_therapy', 'hypertension_and_diabetes', 'heaviness_or_tightness', 'difficulty_finding_shirt_that_fits', 'extra_supplementation', 'dash_score'}"
        })
    
    clf = load("/mnt/access/obj/classifier.mdl")
    prediction = clf.predict_proba(pd.DataFrame(
        columns=[
            'Hormonal therapy',
            'Hypertension & diabetes',
            'SS1: Heaviness/tightness',
            'SS2: Hardness/ difficulty finding shirts that fits',
            'Extra supplementation',
            'DASH score',
            'Children',
            'BMI',
            'Waist to hip datio',
            'BC receptor',
            'Types of surgery',
            'Number of lymph nodes removed'
        ],
        data=[[
            event['children'],
            event['bmi'], 
            event['waist_to_hip_ratio'], 
            event['bc_receptor'], 
            event['types_of_surgery'], 
            event['num_lymph_nodes_removed'], 
            event['hormonal_therapy'],
            event['hypertension_and_diabetes'],
            event['heaviness_or_tightness'],
            event['difficulty_finding_shirt_that_fits'],
            event['extra_supplementation'],
            event['dash_score']
        ]]
    ))[0, 0]
    return json.dumps({
        "status": 200,
        "prediction":prediction
    })