import os
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
import pandas as pd
import json
from services.ResumeInfoExtraction import ResumeInfoExtraction
from services.JobInfoExtraction import JobInfoExtraction
from source.schemas.resumeextracted import ResumeExtractedModel
from source.schemas.jobextracted import JobExtractedModel
import ast
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ast
#import openai
import time
import seaborn as sns
import matplotlib.pyplot as plt
import json
import warnings 
import logging
import main
import os
import json
import pandas as pd
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ast
#import openai
import time
import seaborn as sns
import matplotlib.pyplot as plt
import json
logging.getLogger('pypdf').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Get the absolute path of the root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Paths to your pattern files
degrees_patterns_path = os.path.join(ROOT_DIR, 'matching_algo_internal', 'Resources', 'data', 'degrees.jsonl')
majors_patterns_path = os.path.join(ROOT_DIR, 'matching_algo_internal', 'Resources', 'data', 'majors.jsonl')
skills_patterns_path = os.path.join(ROOT_DIR, 'matching_algo_internal','Resources', 'data', 'skills.jsonl')



def get_resumes(directory):
    
    def extract_pdf(path):
        reader = PdfReader(path)
        number_of_pages = len(reader.pages)
        text = ""
        for i in range(number_of_pages):
            page = reader.pages[i]
            text += page.extract_text()
        return text
    
    dic = {}
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path) and filename.endswith(".pdf"):
            name = filename.strip(".pdf")
            resume_text = extract_pdf(file_path)
            dic[name] = [resume_text]
    
    df = pd.DataFrame(dic).T
    df.reset_index(inplace=True)
    df.rename(columns={"index": "name", 0:"raw"}, inplace=True)
    
    return df

def transform_dataframe_to_json(dataframe):

    # transforms the dataframe into json
    result = dataframe.to_json(orient="records")
    parsed = json.loads(result)
    json_data = json.dumps(parsed, indent=4)

    return json_data


def resume_extraction(resume):
    jobs = resume
    names = transform_dataframe_to_json(jobs[["name"]])
    job_extraction = ResumeInfoExtraction(skills_patterns_path, majors_patterns_path, degrees_patterns_path, jobs, names)
    jobs = job_extraction.extract_entities(jobs)
    for i, row in jobs.iterrows():
        name = row["name"]
        degrees = jobs.loc[i, 'Degrees']
        maximum_degree_level = jobs.loc[i, 'Maximum degree level']
        acceptable_majors = jobs.loc[i, 'Acceptable majors']
        skills = jobs.loc[i, 'Skills']
        

        job_extracted = ResumeExtractedModel(maximum_degree_level=maximum_degree_level if maximum_degree_level else '',
                                          acceptable_majors=acceptable_majors if acceptable_majors else [],
                                          skills=skills if skills else [],
                                          name=name if name else '',
                                          degrees=degrees if degrees else [])
        job_extracted = jsonable_encoder(job_extracted)
    jobs_json = transform_dataframe_to_json(jobs)
    
    return jobs_json


def job_info_extraction(jobs):
    job_extraction = JobInfoExtraction(skills_patterns_path, majors_patterns_path, degrees_patterns_path, jobs)
    jobs = job_extraction.extract_entities(jobs)
    for i, row in jobs.iterrows():
        minimum_degree_level = jobs['Minimum degree level'][i]
        acceptable_majors = jobs['Acceptable majors'][i]
        skills = jobs['Skills'][i]

        job_extracted = JobExtractedModel(minimum_degree_level=minimum_degree_level if minimum_degree_level else '',
                                          acceptable_majors=acceptable_majors if acceptable_majors else [],
                                          skills=skills if skills else [])
        job_extracted = jsonable_encoder(job_extracted)
    jobs_json = transform_dataframe_to_json(jobs)
    return jobs_json

def calc_similarity(applicant_df, job_df):
    """"Calculate cosine simlarity based on BERT embeddings of skills"""

    def semantic_similarity_sbert_base_v2(job,resume):
        """calculate similarity with SBERT all-mpnet-base-v2"""
        model = SentenceTransformer('all-mpnet-base-v2')
        #Encoding:
        score = 0
        sen = job+resume
        sen_embeddings = model.encode(sen)
        for i in range(len(job)):
            if job[i] in resume:
                score += 1
            else:
                max_cosine_sim = max(cosine_similarity([sen_embeddings[i]],sen_embeddings[len(job):])[0]) 
                if max_cosine_sim >= 0.4:
                    score += max_cosine_sim
        score = score/len(job)  
        return round(score,3)
    
    columns = ['applicant', 'job_id', 'all-mpnet-base-v2_score']
    matching_dataframe = pd.DataFrame(columns=columns)
    
    for job_index in range(job_df.shape[0]):
        columns = ['applicant', 'job_id', 'all-mpnet-base-v2_score']
        matching_dataframe = pd.DataFrame(columns=columns)
        ranking_dataframe = pd.DataFrame(columns=columns)
        
        matching_data = []
        
        for applicant_id in range(applicant_df.shape[0]):
            matching_dataframe_job = {
                "applicant": applicant_df.iloc[applicant_id, 0],
                "job_id": job_index,
                "all-mpnet-base-v2_score": semantic_similarity_sbert_base_v2(job_df['Skills'][job_index], applicant_df['Skills'][applicant_id])
            }
            matching_data.append(matching_dataframe_job)
        
        matching_dataframe = pd.concat([matching_dataframe, pd.DataFrame(matching_data)], ignore_index=True)
    matching_dataframe['rank'] = matching_dataframe['all-mpnet-base-v2_score'].rank(ascending=False)
    return matching_dataframe

if __name__ == "__main__":
    # Create DF for resumes
    df = get_resumes("resumes")
    res = resume_extraction(df)
    df = pd.read_json(res)
    print(df)

    # Create DF for jobs
    # Create the full path to the 'description.txt' file
    description_file_path = os.path.join(ROOT_DIR, 'job_descriptions', 'description.txt')
    with open(description_file_path, 'r') as file:
        job_description = file.read()


    job_description = [job_description]
    df2 = pd.DataFrame(job_description, columns=["raw"])
    res = job_info_extraction(df2)
    df2 = pd.read_json(res)
    