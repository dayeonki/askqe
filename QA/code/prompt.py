qa_prompt = """Task: You will be given an English sentence and a list of relevant questions. Your goal is to generate a list of answers to the questions based on the sentence. Output only the list of answers in Python list format without giving any additional explanation. Do not output as code format (```python```).

*** Example Starts ***
Sentence: and does this pain move from your chest?
Questions: ["What moves from your chest?", "Where does the pain move from?"]
Answers: ["The pain", "Your chest"]

Sentence: Diabetes mellitus (784, 10.9%), chronic lung disease (656, 9.2%), and cardiovascular disease (647, 9.0%) were the most frequently reported conditions among all cases.
Questions: ["What were the most frequently reported conditions among all cases?", "Which conditions were reported with a frequency of 10.9%, 9.2%, and 9.0%, respectively?", "What percentage of cases reported diabetes mellitus?", "What percentage of cases reported chronic lung disease?", "What percentage of cases reported cardiovascular disease?"]
Answers: ["Diabetes mellitus, chronic lung disease, and cardiovascular disease", "Diabetes mellitus (10.9%), chronic lung disease (9.2%), and cardiovascular disease (9.0%)", "10.9%", "9.2%", "9.0%"]
*** Example Ends ***

Sentence: {{sentence}}
Questions: {{questions}}
Answers: """