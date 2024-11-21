import os
import pandas as pd
from services.ResumeInfoExtraction import ResumeInfoExtraction
from services.JobInfoExtraction import JobInfoExtraction
from source.schemas.resumeextracted import ResumeExtractedModel # Let's reintroduce later on
from source.schemas.jobextracted import JobExtractedModel # Let's reintroduce later on
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
import markdown
import warnings 
import logging
logging.getLogger('pypdf').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Get the absolute path of the root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Paths to your pattern files
skills_patterns_path = os.path.join(ROOT_DIR, 'workproject_matching_algo','Resources', 'data', 'skills.jsonl')


def get_resumes(directory):
    """ Function to parse and extract text from PDFs in a directory """
    
    def extract_pdf(path):
        """ Helper function to extract the text from the PDFs using the PyMuPDF library"""
        try:
            with fitz.open(path) as doc:
                text = ''.join(page.get_text() for page in doc)
            return text
        except Exception as e:
            logging.error(f"Error processing {path}: {e}")
            return ""

    # initialize empty dictionary
    dic = {}
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Extract text from pdf if file is pdf
        if os.path.isfile(file_path) and filename.endswith(".pdf"):
            name = filename.rstrip(".pdf")
            resume_text = extract_pdf(file_path)
            dic[name] = [resume_text]
    
    # Create a pandas dataframe from the parsed PDFs
    df = pd.DataFrame(dic).T
    df.reset_index(inplace=True)
    df.rename(columns={"index": "name", 0: "raw"}, inplace=True)
    
    return df


def resume_extraction(resumes):
    """ function to extract the relevant skills from a resume """
    names = resumes[["name"]]
    resume_extraction = ResumeInfoExtraction(skills_patterns_path, names)
    resumes_df = resume_extraction.extract_entities(resumes)
    return resumes_df


def job_info_extraction(jobs):
    """ function to extract the relevant skills from a job description """
    job_extraction = JobInfoExtraction(skills_patterns_path)
    job_df = job_extraction.extract_entities(jobs)
    return job_df


def calc_similarity(applicant_df, job_df, N=3):
    """"Calculate cosine simlarity based on BERT embeddings of skills"""

    def semantic_similarity_sbert_base_v2(job,resume):
        """calculate similarity with SBERT all-mpnet-base-v2"""
        model = SentenceTransformer('all-mpnet-base-v2')
        model.eval()
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
    matching_dataframe['interview_status'] = matching_dataframe['rank'].apply(lambda x: 'Selected' if x <= N else 'Not Selected')
    return matching_dataframe

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def tailored_questions(api_key,  applicants, required_skills, model="gpt-4o-mini"):
    """ function to create tailored interview questions with openai api """
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful recruiting assistant. We have a list of candidates we want to interview for a job and we want to tailor interview questions to their skills. Note: to avoid bias we want the applicants to recieive the same questions"}, # <-- This is the system message that provides context to the model
        {"role": "user", "content": f"Hello! Based on the following candidates: {applicants}, could you make a list of 5 interview questions for all of them based on their total pool of skills and how it relates to the skills required of the job - here: {required_skills} "}  # <-- This is the user message for which the model will generate a response
    ]
    )

    markdown_output = completion.choices[0].message.content
    html_output = markdown.markdown(markdown_output)  # Convert markdown to HTML

    return html_output

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def bespoke_apologies(api_key,  applicants, required_skills, model="gpt-4o-mini"):
    """ function to create bespoke apology letters with openai api """
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful recruiting assistant. We have a list of candidates for a job, but unfortunately none of them made it to the first round of interviews."}, # <-- This is the system message that provides context to the model
        {"role": "user", "content": f"""Hello! Based on the following candidates: {applicants}, could you make a bespoke aplogy letter to each of them and explain that their skills were not a 
        prefect match with the required skills here:{required_skills}. For each of the applicants, please also provide them with some resources to improve the skills in which they are lacking so they have better chances in the next round of recruiting """}  # <-- This is the user message for which the model will generate a response
    ]
    )

    markdown_output = completion.choices[0].message.content
    html_output = markdown.markdown(markdown_output)  # Convert markdown to HTML

    return html_output


######################################################################################
###                                  SCRIPT                                        ###
######################################################################################
import time # for benchmarking during code optimization


def main(open_ai=False):
    t0 = time.time()
    # Create DataFrame for resumes
    df_resumes = get_resumes("resumes")
    df_resumes = resume_extraction(df_resumes)
    print(df_resumes[["name", "Skills"]])

    # Create DataFrame for jobs
    description_file_path = os.path.join(ROOT_DIR, 'workproject_matching_algo', 'job_descriptions', 'job1.txt')
    with open(description_file_path, 'r') as file:
        job_description = file.read()
    df_jobs = pd.DataFrame([job_description], columns=["raw"])
    df_jobs = job_info_extraction(df_jobs)
    print(df_jobs)

    # Conduct Similarity Analysis
    analysis_data_df = calc_similarity(df_resumes, df_jobs, parallel=True)
    print(analysis_data_df.sort_values("rank", ascending=True ))

    t1 = time.time()
    dt = t1 - t0
    print(f"dt: {dt*1000:.2f}ms")

    if open_ai:    
        # Set the API key and model name
        MODEL="gpt-4o-mini"
        api_key=os.getenv("OPENAI_API_KEY", "<your OpenAI API key if not set as an env var>")

        # Create tailored interview questions
        tailored_questions = tailored_questions(api_key, df_resumes, df_jobs['Skills'], model=MODEL)
        print(tailored_questions)

        # Create bespoke apologies
        bespoke_apologies = bespoke_apologies(api_key, df_resumes, df_jobs['Skills'], model=MODEL)
        print(bespoke_apologies)


if __name__ == "__main__":
    main(open_ai=False)
    
