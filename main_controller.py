# import modules
import logging

import pandas as pd

import api_mod
import data_org_mod
import file_io_mod
import output_mod
from api_mod import OpenAIAPIClient

test_bank_name = 'Chapter_15.txt'
chapter_name = 'ch15.pdf'

# set up logging 
logging.basicConfig(filename='running-output.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # file i/o operations
    question_material = file_io_mod.read_txt(test_bank_name)
    chapter_material = file_io_mod.read_pdf(chapter_name)

    # split questions and chapter texts
    question_blocks = data_org_mod.split_questions(question_material)
    subchapter_blocks = data_org_mod.split_subchapters(chapter_material)
    # TODO: verify that inputs look correct and are formatted well

    # create dataframe for question blocks
    df_Data = []
    for question_block in question_blocks:
        question, correct_answer, learning_objective = data_org_mod.extract_info_from_block(question_block)
        df_Data.append({
            'Question' : question,
            'Correct_Answer' : correct_answer,
            'Learning_Objective' : learning_objective
        })
    df_questions = pd.DataFrame(df_Data)

    # create dataframe for subchapter_blocks
    df_data = []
    for subchapter_block in subchapter_blocks:
        learning_objective, content = data_org_mod.extract_subchapter_info(subchapter_block)
        df_data.append({
            'Learning_Objective' : learning_objective,
            'Content' : content
        })
    df_subchapters = pd.DataFrame(df_data)
    #print(df_subchapters.head())
    #print(df_questions.head())

    #print(question_blocks[2])
    #print(subchapter_blocks[2])
    
    # perform processes and tasks


    # test extract from block


    results = api_mod.run_process1(df_questions, df_subchapters)
    print(results.head())
    results.to_excel('output.xlsx', index=True)


    # output to .csv

if __name__ == "__main__":
    main()