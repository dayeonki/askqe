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

prompts = {
    "atomic": nli,
}


atomic_fact_prompt = """Task: You will be given an English sentence. Your goal is to identify a list of atomic facts from the sentence. Atomic fact is a short sentence conveying one piece of information. Output the list of atomic facts in Python list format without giving any additional explanation.

*** Example Starts ***
Sentence: It is not yet known whether the severity or level of control of underlying health conditions affects the risk for severe disease associated with COVID-19.
Atomic facts: ['It is not yet known whether the severity of underlying health conditions affects the risk for severe disease associated with COVID-19.', 'It is not yet known whether the level of control of underlying health conditions affects the risk for severe disease associated with COVID-19.']

Snetence: The number of accessory proteins and their function is unique depending on the specific coronavirus.
Atomic facts: ['The number of accessory proteins is unique depending on the specific coronavirus.', 'The function of accessory proteins is unique depending on the specific coronavirus.']
*** Example Ends ***

Setence: {{sentence}}
Atomic facts: """