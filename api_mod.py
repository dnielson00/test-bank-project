import concurrent.futures
import logging
import re
import warnings

import nltk
import openai
import pandas as pd
import requests
import transformers
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from requests.exceptions import HTTPError, ReadTimeout
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from tenacity import (after_log, before_sleep_log, retry,
                      retry_if_exception_type, stop_after_attempt, wait_fixed)
from transformers import GPT2Tokenizer

# Ensure you have the stopwords dataset downloaded
nltk.download('stopwords')
nltk.download('punkt')

class OpenAIAPIClient:
    def __init__(self, api_key, engine="gpt-3.5-turbo"):
        self.api_key = api_key
        self.engine = engine

    @retry(retry=retry_if_exception_type((HTTPError, openai.error.APIError, openai.error.OpenAIError, openai.error.RateLimitError, TimeoutError)), stop=stop_after_attempt(5), reraise=True, before_sleep=before_sleep_log(logging, logging.INFO))
    def query(self, prompt, tokens=50):
        """
        Query the OpenAI api with a dynamic prompt.
        :param prompt: prompt to send to the API
        :param tokens: the max number of tokens for the API to return.
        : return: the API's response.
        """
        openai.api_key = self.api_key
        retries = 0
        max_retries = 5
        retry_delay = 1
        backoff_factor = 1.5
        prompt_token_count = calculate_token_count(prompt)
        max_prompt_tokens = 4097 - tokens

        # log initial prompt token length
        logging.info(f"initial prompt token length: {prompt_token_count}")
        if prompt_token_count > max_prompt_tokens:
            #print(f"prompt too long: {prompt_token_count} tokens. Reducing..")
            prompt = reduce_tokens(prompt, target_token_count=max_prompt_tokens)
            #print(f"reduced to {calculate_token_count(prompt)} tokens")
        # log final prompt token length
        logging.info(f"final prompt token length: {calculate_token_count(prompt)}")

        #while retries < max_retries: # retry mechanism in case there are issues with api calls for whatever reason
        try:    
            # Query GPT-3.5-turbo
            response = openai.ChatCompletion.create(
                    model=self.engine,
                    max_tokens = tokens,
                    messages=[
                    {"role": "system", "content": "You are a helpful assistant. Respond with 'yes', 'no', or 'unknown'. Also, provide a confidence score from 0 to 1 of your assessment in the format 'Confidence: x.x'."},
                    {"role": "user", "content": prompt}
                ],
            )
                
            # Extract and return the text of the response
            answer = response['choices'][0]['message']['content']
            return answer
            #! may need to delete below exception if it doesnt play nice
            #except openai.error.Timeout as e: # this would be redundant if the openai.error.OpenAIError exception is the one that triggers instead of this one
             #   retries += 1
              #  print(f"OpenAI Timeout occurred: retrying {retries}/{max_retries} after {retry_delay} seconds.")
               #retry_delay *= backoff_factor  # Increase the delay for the next retry
            
        except TimeoutError as e:
            logging.info(f"Timeout occured: {e}. Retrying..")
            raise
        except openai.error.OpenAIError as e:
            if hasattr(e, 'response') and 'Retry-After' in e.response.headers:
                retry_after = int(e.response.headers['Retry-After'])
                logging.info(f"Retrying after {retry_after} seconds due to rate limit.")
                time.sleep(retry_after)
                raise
            else:
                logging.error(f"an OpenAI API error occured: {str(e)}")
                raise e
        except Exception as e:
            # Handle other exceptions that may occur
            print(f"An error occurred: {e}")
            return f"an error occured: {e}"        
    # todo: summarize input text until it matches the number of tokens specified in query
    #def summarize_recursively():

    # query functions
    def validate_answer(self, question, correct_answer, subchapter_content):
        prompt = f"Based on the following subchapter content, is the answer to the question correct?\n\nSubchapter Content:\n{reduce_tokens(subchapter_content)}\n\nQuestion: {question}\n\nAnswer: {correct_answer}"
        return self.query(prompt)

    def validate_question(self, question, subchapter_content):
        prompt = f"""
        Can the question be answered with the information provided in the subchapter content? 
        Please answer 'yes' or 'no' and provide a confidence score from 0 to 1.
        - Respond 'yes' ONLY if the information directly answers the question.
        - Respond 'no' if the information does not pertain to the question at all.
        - If the information is not enough to fully answer the question, please write 'unknown' followed by what is missing\n\n
        Subchapter Content:\n{reduce_tokens(subchapter_content)}\n\nQuestion: {question}"""

        response = self.query(prompt, tokens=100)
        # Use regex to extract the confidence score
        confidence_match = re.search(r'Confidence: (\d+\.\d+)', response)
        if confidence_match:
            confidence = float(confidence_match.group(1))
        else:
            confidence = None  # or some default value if confidence is not found
        
        return response, confidence

    def validate_LO(self, question, learning_objective, subchapter_content):
        prompt = f"Based on the following subchapter content, is the learning objective correct for the question?\n\nSubchapter Content:\n{reduce_tokens(subchapter_content)}\n\nQuestion: {question}\n\nLearning Objective: {learning_objective}"
        return self.query(prompt)
    
    def find_appropriate_subchapter(self, question, df_subchapters):
        max_confidence = 0
        best_fit_subchapter = None

        for index, row in df_subchapters.iterrows():
            learning_objective = row['Learning_Objective']
            subchapter_content = row['Content']

            # Query the API to evaluate how well the question fits with this subchapter
            prompt = f"Based on the following subchapter content, how relevant is the question?\n\nSubchapter Content:\n{subchapter_content}\n\nQuestion: {question}"
            response = self.query(prompt)

            # check if response is none
            if response is None:
                print(f"Query returned None for subchapter {index}.")
                continue

            # Parse the API response to get a confidence score (you may need to adjust this part based on the API's output)
            confidence_match = re.search(r'Confidence: (\d+\.\d+)', response)
            if confidence_match:
                confidence_score = float(confidence_match.group(1))
            else:
                confidence_score = None  # or some default value if confidence is not found


            if confidence_score is not None and confidence_score > max_confidence:
                max_confidence = confidence_score
                best_fit_subchapter = learning_objective

        return best_fit_subchapter, max_confidence



def process1(question_num, question, test_bank_answer, learning_objective, df_subchapters, api_client):
    # Initialize variables
    is_LO_correct = False
    correct_subchapter = 'TBD'
    
    # Fetch the relevant subchapter content based on the learning objective
    subchapter_content = df_subchapters[df_subchapters['Learning_Objective'] == learning_objective]['Content'].values[0] if learning_objective in df_subchapters['Learning_Objective'].values else ""

    # First Query: Validate if the question is answerable based on the subchapter content
    start_time = time.perf_counter()
    is_question_answerable, confidence_score = api_client.validate_question(question, subchapter_content)
    logging.info(f"validate_question took {time.perf_counter() - start_time} seconds")
    #f'is the question answerable? --{is_question_answerable}--')
    #print(f'confidence score is: {confidence_score}')

    # TODO: modify this so confidence score can be changed based on the first query
    if confidence_score is not None and confidence_score > 0.7:  # Threshold can be adjusted
        is_LO_correct = True
        correct_subchapter = learning_objective
    
    # Second Query: If confidence is low, find the most appropriate subchapter
    if not is_LO_correct:
        start_time = time.perf_counter()
        correct_subchapter, _ = api_client.find_appropriate_subchapter(question, df_subchapters)
        logging.info(f"fine_appropriate_subchaper took {time.perf_counter() - start_time} seconds")
    
    # Fetch the content for the correct subchapter
    correct_subchapter_content = df_subchapters.get(correct_subchapter, "")
    
    # Third Query: Validate the correct answer
    start_time = time.perf_counter()
    is_answer_correct = api_client.validate_answer(question, test_bank_answer, correct_subchapter_content)
    logging.info(f"vvalidate_answer took {time.perf_counter() - start_time} seconds")
    
    result = {
        'Question Number': question_num,
        'Question': question,
        'Answer on Test Bank': test_bank_answer,
        'Learning_Objective': learning_objective,
        'Is_Answer_Correct': is_answer_correct,
        'Is_Question_Answerable': is_question_answerable,
        'Is_LO_Correct': is_LO_correct,
        'Correct_Subchapter': correct_subchapter
    }

    # convert result to string for logging and log
    result_str = ', '.join(f'{key}: {value}         ' for key, value in result.items())
    logging.info({result_str})

    return result
   
def run_process1(df_questions, df_subchapters):
    api_key = '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'
    api_client = OpenAIAPIClient(api_key)
    results = []
    bucket = TokenBucket(rate=3500/60, capacity=3500) # rate limiting mechanism

 

    for index, row in df_questions.iterrows():
        while not bucket.consume():
            time.sleep(1) # sleep for a while before retrying

        question = row['Question']
        correct_answer = row['Correct_Answer']
        learning_objective = row['Learning_Objective']  # Make sure the column name matches your DataFrame

        print(f"currently working on index {index}")
        
        #if index > 10: 
        #    break

        try:
            start_time = time.perf_counter() # start logging time to identify slower parts of the code
            result = process1(index, question, correct_answer, learning_objective, df_subchapters, api_client)
            logging.info(f"completed analysis of question {index}. Duration: {time.perf_counter() - start_time:.2f} seconds")
        except openai.error.InvalidRequestError as e:
            print({"error": str(e)})
            result = {
                'Question': question,
                'Answer on Test Bank': correct_answer,
                'Learning_Objective': learning_objective,
                'Is_Answer_Correct': 'error',
                'Is_Question_Answerable': 'error',
                'Is_LO_Correct': 'error',
                'Correct_Subchapter': 'error'
            }
        
        results.append(result)


    df_results = pd.DataFrame(results)
    return df_results


import time


class TokenBucket:
    def __init__(self, rate, capacity):
        self.rate = rate  # tokens per second
        self.capacity = capacity  # maximum tokens
        self.tokens = capacity  # initial tokens
        self.last_refill_time = time.time()

    def refill(self):
        now = time.time()
        elapsed = now - self.last_refill_time
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill_time = now

    def consume(self):
        self.refill()
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

def reduce_tokens2(text, target_token_count=4097-100):
    # Assuming 'text' is your text data
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    # Load the GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    sentence_count = len(parser.document.sentences)  # Start with the original number of sentences
    while True:
        summary = summarizer(parser.document, sentences_count=sentence_count)
        summarized_text = ' '.join([str(sentence) for sentence in summary])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tokens = tokenizer.tokenize(summarized_text)
        if len(tokens) < target_token_count:
            return summarized_text  # Exit the loop if the token count is below 4097
        sentence_count -= 1  # Reduce the sentence count by 1 in the next iteration

        if sentence_count == 0:  # Prevent an infinite loop
            print("Unable to reduce text to target token count.")
            break

    raise ValueError("unable to reduce text to tarket token count :(")


def reduce_tokens(text, target_token_count=4097-100):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Define the bounds for binary search
    lower_bound = 0
    upper_bound = len(parser.document.sentences)
    while lower_bound < upper_bound:
        sentence_count = (lower_bound + upper_bound) // 2
        summary = summarizer(parser.document, sentences_count=sentence_count)
        summarized_text = ' '.join([str(sentence) for sentence in summary])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tokens = tokenizer.tokenize(summarized_text)

        if len(tokens) > target_token_count:
            upper_bound = sentence_count - 1  # Too long, decrease upper bound
        else:
            lower_bound = sentence_count + 1  # Short enough, increase lower bound to find longer summary within limit

    # Return the summary that is right under the token limit
    summary = summarizer(parser.document, sentences_count=upper_bound)
    summarized_text = ' '.join([str(sentence) for sentence in summary])
    return summarized_text

def calculate_token_count(text):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return len(GPT2Tokenizer.from_pretrained('gpt2').encode(text))
