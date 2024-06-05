import re


def split_subchapters(text):
    relevant_section = extract_relevant_section(text)
    pattern = re.compile(r'\b\d{1,2}-\d+(?:[a-z])?\s.+?(?=\n\d{1,2}-\d+\s|$|Chapter Review)', re.DOTALL)
    subchapters = re.findall(pattern, relevant_section)
    return subchapters

# helper method that extracts the actual chapters out of the pdf and removes extraneous text
def extract_relevant_section(text):
    # Locate the indices of the start and end keywords
    chapter_start_pattern = re.compile(r'Chapter \d+:')
    start_pattern = re.compile(r'\b\d{1,2}-1 .+?\n')
    end_pattern = re.compile(r'Chapter Review')
    
    chapter_start_match = chapter_start_pattern.search(text)
    start_match = start_pattern.search(text, pos=chapter_start_match.start())
    end_match = end_pattern.search(text)

    if chapter_start_match and start_match and end_match:
        # Extract the relevant section
        start_indx = start_match.start()
        end_indx = end_match.start()
        return text[start_indx:end_indx]
    else:
        print("Start of end keywords not found")
        return None
    
# split each question 
def split_questions(text):
    return re.split(re.compile(r'(?<=\n\n)\d+\.\s'), text)

# function to extract the important information from a question block
def extract_info_from_block(question_block):
    try:
        # use regex to extract relevant information
        pattern = re.compile(r'^(.*?)(?=\n\s*\ta\.)', re.DOTALL)
        question = pattern.search(question_block).group(1).strip()
        correct_answer = re.search(r'ANSWER:\s*\n\t*(.*?)(?=\n\tPOINTS:|\n\tDIFFICULTY:)', question_block, re.DOTALL).group(1).strip()
        learning_objective = re.search(r'LO:\s*([0-9]{1,2}-[0-9]{1,2})', question_block, re.IGNORECASE).group(1)
        #print(question)
        return question, correct_answer, learning_objective
    except AttributeError:
        #TODO: change this to handle short answer questions -- there are relatively few so this is low priority rn
        #print("Skipping a question block due to missing or malformed data.")
        #print(question_block)
        #print("------------------------------------------------------------------")
        return None,None,None
    

def extract_subchapter_info(subchapter_block):
    try: 
        # use regex to extract relevant information
        pattern = re.compile(r'(\d{1,2}-\d{1,2})')
        match = pattern.search(subchapter_block)
        
        if match:
            subchapter_number = match.group(1).strip()
            return subchapter_number, subchapter_block
        else:
            print("No match found in subchapter block.")
            return None, None

    except AttributeError:
        print("Skipping a subchapter block due to missing or malformed data.")
        return None, None