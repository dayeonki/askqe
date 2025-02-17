vanilla = """Task: You will be given an English sentence. Your goal is to generate a list of relevant questions based on the sentence. Output only the list of questions in Python list format without giving any additional explanation.

*** Example Starts ***
Sentence: It is not yet known whether the severity or level of control of underlying health conditions affects the risk for severe disease associated with COVID-19.
Questions: ["What is not yet known?", "What might affect the risk for severe disease associated with COVID-19?", "What is associated with severe disease?", "What conditions are mentioned in the sentence?", "What is the disease mentioned in the sentence?"]

Sentence: The number of accessory proteins and their function is unique depending on the specific coronavirus.
Questions: ["What is unique depending on the specific coronavirus?", "What is unique about the function of accessory proteins?"]
*** Example Ends ***

Sentence: {{sentence}}
Questions: """


nli = """Task: You will be given an English sentence and a list of atomic facts, which are short sentences conveying one piece of information. Your goal is to generate a list of relevant questions based on the sentence. Output the list of questions in Python list format without giving any additional explanation. Do not output as code format (```python```).

*** Example Starts ***
Sentence: It is not yet known whether the severity or level of control of underlying health conditions affects the risk for severe disease associated with COVID-19.
Atomic facts: ['It is not yet known whether the severity of underlying health conditions affects the risk for severe disease associated with COVID-19.', 'It is not yet known whether the level of control of underlying health conditions affects the risk for severe disease associated with COVID-19.']
Questions: ["Is it known whether the severity of underlying health conditions affects the risk for severe disease associated with COVID-19?", "Is it known whether the level of control of underlying health conditions affects the risk for severe disease associated with COVID-19?"]

Sentence: The number of accessory proteins and their function is unique depending on the specific coronavirus.
Atomic facts: ['The number of accessory proteins is unique depending on the specific coronavirus.', 'The function of accessory proteins is unique depending on the specific coronavirus.']
Questions: ["What is unique depending on the specific coronavirus?", "What is unique about the function of accessory proteins?"]
*** Example Ends ***

Sentence: {{sentence}}
Atomic facts: {{atomic_facts}}
Questions: """


srl = """Task: You will be given an English sentence and a dictionary of semantic roles in the sentence. Your goal is to generate a list of relevant questions based on the sentence. Output the list of questions in Python list format without giving any additional explanation. Do not output as code format (```python```).

*** Example Starts ***
Sentence: It is not yet known whether the severity or level of control of underlying health conditions affects the risk for severe disease associated with COVID-19.
Semantic roles: {'Verb1': {'Verb': 'known', 'ARG1': 'It', 'TMP': 'not yet', 'ARG2': 'whether the severity or level of control of underlying health conditions affects the risk for severe disease associated with COVID-19'}, 'Verb2': {'Verb': 'affects', 'ARG0': 'the severity or level of control of underlying health conditions', 'ARG1': 'the risk for severe disease associated with COVID-19'}}
Questions: ["What is not yet known?","When is it not yet known?","What is being questioned in terms of its effect on the risk for severe disease associated with COVID-19?","What affects the risk for severe disease associated with COVID-19?","What is the risk associated with COVID-19?"]

Sentence: The number of accessory proteins and their function is unique depending on the specific coronavirus.
Semantic roles: {'Verb': 'is', 'ARG1': 'The number of accessory proteins and their function', 'MNR': 'unique', 'TMP': 'depending on the specific coronavirus'}
Questions: ["What is unique depending on the specific coronavirus?", "How is the number of accessory proteins and their function described?", "When is the uniqueness of the number of accessory proteins and their function determined?"]
*** Example Ends ***

Sentence: {{sentence}}
Semantic roles: {{semantic_roles}}
Questions: """


prompts = {
    "vanilla": vanilla,
    "atomic": nli,
    "semantic": srl
}