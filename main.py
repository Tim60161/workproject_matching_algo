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
#skills_patterns_path = os.path.join(ROOT_DIR, 'workproject_matching_algo Kopie','Resources', 'data', 'skills.jsonl')

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


def calc_similarity(applicant_df, job_df, N=3, parallel=False):
    """Calculate cosine similarity based on MPNET embeddings of combined skills."""

    # Initialize the model once outside the loop for efficiency
    model = SentenceTransformer('all-mpnet-base-v2')
    model.max_seq_length = 75
    model.tokenizer.padding_side="right"
    model.eval()

    def add_eos(input_examples):
        """ helper function to add special tokens between each skills"""
        input_examples = [input_example + model.tokenizer.eos_token for input_example in input_examples]
        return input_examples

        # Precompute job embeddings
    job_df['Skills_Text'] = job_df['Skills'].apply(add_eos)
    job_df['Skills_Text'] = job_df['Skills_Text'].apply(lambda x: ' '.join(sorted(set(x))) if isinstance(x, list) else '')
    job_embeddings = model.encode(
        job_df['Skills_Text'].tolist())
    # Precompute applicant embeddings
    applicant_df['Skills_Text'] = applicant_df['Skills'].apply(add_eos)
    applicant_df['Skills_Text'] = applicant_df['Skills_Text'].apply(lambda x: ' '.join(sorted(set(x))) if isinstance(x, list) else '')
    applicant_embeddings = model.encode(
        applicant_df['Skills_Text'].tolist(),
        batch_size=32,
        num_workers=os.cpu_count() // 2 if parallel else 0,
        show_progress_bar=False
    )

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(job_embeddings, applicant_embeddings)

    # Create a DataFrame from the similarity matrix
    similarity_df = pd.DataFrame(similarity_matrix.T, index=applicant_df['name'], columns=job_df.index)
    similarity_df = similarity_df.reset_index().melt(id_vars='name', var_name='job_id', value_name='similarity_score')
    similarity_df['rank'] = similarity_df.groupby('job_id')['similarity_score'].rank(ascending=False)
    similarity_df['interview_status'] = similarity_df['rank'].apply(lambda x: 'Selected' if x <= N else 'Not Selected')

    return similarity_df

def calc_similarity_sbs(applicant_df, job_df):
    """Calculate cosine similarity based on BERT embeddings of skills (skill-by-skill)."""

    def semantic_similarity_sbert_base_v2(job, resume):
        """Calculate similarity with SBERT all-mpnet-base-v2."""
        model = SentenceTransformer('all-mpnet-base-v2')
        model.eval()
        score = 0
        sen = job + resume
        sen_embeddings = model.encode(sen)
        for i in range(len(job)):
            if job[i] in resume:
                score += 1
            else:
                max_cosine_sim = max(cosine_similarity([sen_embeddings[i]], sen_embeddings[len(job):])[0])
                if max_cosine_sim >= 0.4:
                    score += max_cosine_sim
        score = score / len(job)
        return round(score, 3)

    # Prepare a DataFrame to store results
    matching_dataframe = []

    # Loop through each job in the job_df
    for job_index in range(len(job_df)):
        job_skills = job_df['Skills'].iloc[job_index]  # Use iloc for positional indexing

        # Loop through each applicant in the applicant_df
        for applicant_id in range(len(applicant_df)):
            applicant_skills = applicant_df['Skills'].iloc[applicant_id]  # Use iloc for positional indexing
            applicant_name = applicant_df['name'].iloc[applicant_id]  # Ensure correct column access

            # Compute similarity score
            score = semantic_similarity_sbert_base_v2(job_skills, applicant_skills)

            # Append result to the DataFrame
            matching_dataframe.append({
                "applicant": applicant_name,
                "job_id": job_index,
                "all-mpnet-base-v2_score": score
            })

    # Create a DataFrame from results
    matching_dataframe = pd.DataFrame(matching_dataframe)

    # Add rank based on similarity score
    matching_dataframe['rank'] = matching_dataframe['all-mpnet-base-v2_score'].rank(ascending=False)
    return matching_dataframe

def calc_similarity_sbs_all_MiniLM_L6_v2(applicant_df, job_df):
    """Calculate cosine similarity based on BERT embeddings of skills (skill-by-skill)."""

    def semantic_similarity_all_MiniLM_L6_v2(job, resume):
        """Calculate similarity with all-MiniLM-L6-v2."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        model.eval()
        score = 0
        sen = job + resume
        sen_embeddings = model.encode(sen)
        for i in range(len(job)):
            if job[i] in resume:
                score += 1
            else:
                max_cosine_sim = max(cosine_similarity([sen_embeddings[i]], sen_embeddings[len(job):])[0])
                if max_cosine_sim >= 0.4:
                    score += max_cosine_sim
        score = score / len(job)
        return round(score, 3)

    # Prepare a DataFrame to store results
    matching_dataframe = []

    # Loop through each job in the job_df
    for job_index in range(len(job_df)):
        job_skills = job_df['Skills'].iloc[job_index]  # Use iloc for positional indexing

        # Loop through each applicant in the applicant_df
        for applicant_id in range(len(applicant_df)):
            applicant_skills = applicant_df['Skills'].iloc[applicant_id]  # Use iloc for positional indexing
            applicant_name = applicant_df['name'].iloc[applicant_id]  # Ensure correct column access

            # Compute similarity score
            score = semantic_similarity_all_MiniLM_L6_v2(job_skills, applicant_skills)

            # Append result to the DataFrame
            matching_dataframe.append({
                "applicant": applicant_name,
                "job_id": job_index,
                "all-MiniLM-L6-v2_score": score
            })

    # Create a DataFrame from results
    matching_dataframe = pd.DataFrame(matching_dataframe)

    # Add rank based on similarity score
    matching_dataframe['rank'] = matching_dataframe['all-MiniLM-L6-v2_score'].rank(ascending=False)
    return matching_dataframe

def calc_similarity_sbs_NV_Embed_v2(applicant_df, job_df):
    """Calculate cosine similarity based on NV-Embed-v2 embeddings of skills (skill-by-skill)."""

    def semantic_similarity_NV_Embed_v2(job, resume):
        """Calculate similarity with NV-Embed-v2."""
        model = SentenceTransformer('nvidia/NV-Embed-v2', use_auth_token='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', trust_remote_code=True)  # Add trust_remote_code=True
        model.eval()
        score = 0
        sen = job + resume
        sen_embeddings = model.encode(sen)
        for i in range(len(job)):
            if job[i] in resume:
                score += 1
            else:
                max_cosine_sim = max(cosine_similarity([sen_embeddings[i]], sen_embeddings[len(job):])[0])
                if max_cosine_sim >= 0.4:
                    score += max_cosine_sim
        score = score / len(job)
        return round(score, 3)

    # Prepare a DataFrame to store results
    matching_dataframe = []

    # Loop through each job in the job_df
    for job_index in range(len(job_df)):
        job_skills = job_df['Skills'].iloc[job_index]  # Use iloc for positional indexing

        # Loop through each applicant in the applicant_df
        for applicant_id in range(len(applicant_df)):
            applicant_skills = applicant_df['Skills'].iloc[applicant_id]  # Use iloc for positional indexing
            applicant_name = applicant_df['name'].iloc[applicant_id]  # Ensure correct column access

            # Compute similarity score
            score = semantic_similarity_NV_Embed_v2(job_skills, applicant_skills)

            # Append result to the DataFrame
            matching_dataframe.append({
                "applicant": applicant_name,
                "job_id": job_index,
                "NV-Embed-v2_score": score
            })

    # Create a DataFrame from results
    matching_dataframe = pd.DataFrame(matching_dataframe)

    # Add rank based on similarity score
    matching_dataframe['rank'] = matching_dataframe['NV-Embed-v2_score'].rank(ascending=False)
    return matching_dataframe

def calc_similarity_sbs_BinGSE_MetaLlama_3_8B_Instruct(applicant_df, job_df, tokenizer, model):
    """Calculate cosine similarity based on BinGSE-Meta-Llama-3-8B-Instruct embeddings of skills (skill-by-skill)."""

    def semantic_similarity_BinGSE_MetaLlama_3_8B_Instruct(job, resume, tokenizer, model):
        """Calculate similarity with BinGSE-Meta-Llama-3-8B-Instruct."""
        import torch  # Import torch locally if needed
        max_length = 75  # Define a reasonable max length for truncation
        # Tokenize job and resume text
        job_tokens = tokenizer(job, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        resume_tokens = tokenizer(resume, return_tensors="pt", truncation=True, padding=True, max_length=max_length)

        # Move tokens to the appropriate device and ensure correct tensor type
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        job_tokens = {key: value.to(device).long() if key == "input_ids" else value.to(device) for key, value in job_tokens.items()}
        resume_tokens = {key: value.to(device).long() if key == "input_ids" else value.to(device) for key, value in resume_tokens.items()}

        # Generate embeddings
        with torch.no_grad():
            job_embedding = model(**job_tokens).last_hidden_state.mean(dim=1).cpu().numpy()
            resume_embedding = model(**resume_tokens).last_hidden_state.mean(dim=1).cpu().numpy()

        # Calculate cosine similarity
        similarity = cosine_similarity(job_embedding, resume_embedding)[0][0]
        return round(similarity, 3)

    # Prepare a DataFrame to store results
    matching_dataframe = []

    # Loop through each job in the job_df
    for job_index in range(len(job_df)):
        job_skills = job_df['Skills'].iloc[job_index]  # Use iloc for positional indexing

        # Loop through each applicant in the applicant_df
        for applicant_id in range(len(applicant_df)):
            applicant_skills = applicant_df['Skills'].iloc[applicant_id]  # Use iloc for positional indexing
            applicant_name = applicant_df['name'].iloc[applicant_id]  # Ensure correct column access

            # Compute similarity score
            score = semantic_similarity_BinGSE_MetaLlama_3_8B_Instruct(
                " ".join(job_skills), " ".join(applicant_skills), tokenizer, model
            )

            # Append result to the DataFrame
            matching_dataframe.append({
                "applicant": applicant_name,
                "job_id": job_index,
                "BinGSE-Meta-Llama-3-8B-Instruct_score": score
            })

    # Create a DataFrame from results
    matching_dataframe = pd.DataFrame(matching_dataframe)

    # Add rank based on similarity score
    matching_dataframe['rank'] = matching_dataframe['BinGSE-Meta-Llama-3-8B-Instruct_score'].rank(ascending=False)
    return matching_dataframe

def calc_similarity_sbs_VoyageLarge2Instruct(applicant_df, job_df, tokenizer, model):
    """Calculate cosine similarity based on voyage-large-2-instruct embeddings of skills (skill-by-skill)."""

    def semantic_similarity_VoyageLarge2Instruct(job, resume, tokenizer, model):
        """Calculate similarity with voyage-large-2-instruct."""
        import torch  # Import torch locally if needed
        max_length = 75  # Define a reasonable max length for truncation
        # Tokenize job and resume text
        job_tokens = tokenizer(job, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        resume_tokens = tokenizer(resume, return_tensors="pt", truncation=True, padding=True, max_length=max_length)

        # Move tokens to the appropriate device and ensure correct tensor type
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        job_tokens = {key: value.to(device) for key, value in job_tokens.items()}
        resume_tokens = {key: value.to(device) for key, value in resume_tokens.items()}

        # Generate embeddings
        with torch.no_grad():
            job_embedding = model(**job_tokens).last_hidden_state.mean(dim=1).cpu().numpy()
            resume_embedding = model(**resume_tokens).last_hidden_state.mean(dim=1).cpu().numpy()

        # Calculate cosine similarity
        similarity = cosine_similarity(job_embedding, resume_embedding)[0][0]
        return round(similarity, 3)

    # Prepare a DataFrame to store results
    matching_dataframe = []

    # Loop through each job in the job_df
    for job_index in range(len(job_df)):
        job_skills = job_df['Skills'].iloc[job_index]  # Use iloc for positional indexing

        # Loop through each applicant in the applicant_df
        for applicant_id in range(len(applicant_df)):
            applicant_skills = applicant_df['Skills'].iloc[applicant_id]  # Use iloc for positional indexing
            applicant_name = applicant_df['name'].iloc[applicant_id]  # Ensure correct column access

            # Compute similarity score
            score = semantic_similarity_VoyageLarge2Instruct(
                " ".join(job_skills), " ".join(applicant_skills), tokenizer, model
            )

            # Append result to the DataFrame
            matching_dataframe.append({
                "applicant": applicant_name,
                "job_id": job_index,
                "VoyageLarge2Instruct_score": score
            })

    # Create a DataFrame from results
    matching_dataframe = pd.DataFrame(matching_dataframe)

    # Add rank based on similarity score
    matching_dataframe['rank'] = matching_dataframe['VoyageLarge2Instruct_score'].rank(ascending=False)
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
###                                  SCRIPTS                                        ###
######################################################################################
import time # for benchmarking during code optimization


def main(open_ai=False):
    t0 = time.time()
    # Create DataFrame for resumes
    df_resumes = get_resumes("resumes")
    df_resumes = resume_extraction(df_resumes)
    print(df_resumes[["name", "Skills"]])

    # Create DataFrame for jobs
    #description_file_path = os.path.join(ROOT_DIR, 'workproject_matching_algo', 'job_descriptions', 'job1.txt')
    description_file_path = os.path.join(ROOT_DIR, 'workproject_matching_algo', 'job_descriptions', 'job3.txt')
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