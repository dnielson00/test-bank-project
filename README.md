# test-bank-project
this project was created to assist with editing and quality control for a textbook. it was later abandoned due to scope issues and a lower accuracy than was required


# Description of the Problem
I was given the task of sorting though the test bank for a chapter that was being edited for each chapter. This basically meant that there were questions about each subchapter within each test bank. Each test bank had to be verified for several things, including but not limited to:
- each question must be answerable within the chapter
- each question must have the correct answer available as an option, and it must be correctly labeled as the correct answer in the key
- each question much have the correct formatting, with the correct learning objective as well as subchapter that the question could be found in
- each question should comply with accessibility standards. the most relevant one for me was to ensure that there were no fill-in-the-blank questions
  
With there being approximately 20 chapters, with associated test banks containing 200+ questions each, this was a very time-consuming task with a high potential for human error due to the repetitive nature of the tasks. 

## My goal was to automate the initial screening of each test bank, highlighting areas where human review is necessary.
The idea was to save time and decrease the potential for error inherent to the project


# Note: the code was written and evaluated, and was found to be quite inconsistent. It was never used for the test bank project in any capacity that saw the light of day
I'd like to revist some application of this some day, especially as LMMs and similar technologies improve and may be able to give better results

# How it Works
there is a main controller file that orchestrates everything, if you are reading through the code base, start there. It reads two separate files, the test bank itself as well as the chapter it is trying to verify with. the test bank was divided into blocks for each question, as well as blocks for each subchapter (e.g. chapter 1 subchapter 1, chapter 1 subchapter 2, etc). They then get passed to other functions for processing, which includes openai's gpt-3.5-turbo. 
