question_categorize = """Task: You will be given a question. Your goal is to annotate the question type. The question type reflects the nature of the question. It is NOT determined by the interrogative word of the question. There are 10 question types in total. The definition for each type is shown in the following, along with examples per question type. During annotation, you can label two most-confident types when no clear decision can be made for the most probable type. Output you answer in Python list format without giving any additional explanation.

*** Question Type Starts ***
1. Verification: Asking for the truthfulness of an event or a concept.
- Is Michael Jackson an African American?
- Could stress, anxiety, or worry cause cholesterol levels to rise?

2. Disjunctive: Asking for the true one given multiple events or concepts, where comparison among options is not needed.
- Is Michael Jackson an African American or Latino?
- When you get a spray-on tan does someone put it on you or does a machine do it?

3. Concept: Asking for a definition of an event or a concept.
- Who said the sun never sets on the British empire?
- Where do dolphins have hair at?

4. Extent: Asking for the extent or quantity of an event or a concept.
- How long does gum stay in your system?
- To what extent is the Renewable Fuel Standard accurate nationwide?

5. Example: Asking for example(s) or instance(s) of an event or a concept.
- What are some examples to support or contradict this?
- What countries/regions throughout the world do not celebrate the Christmas holidays?

6. Comparison: Asking for comparison among multiple events or concepts.
- What is the best tinted facial moisturizer?
- In what hilariously inaccurate ways is your job/career portrayed on television or in movies?

7. Cause: Asking for the cause or reason for an event or a concept.
- Why are parents strick on girls than boys?
- What makes nerve agents like 'Novichok' so hard to produce and why can only a handful of laboratories create them?

8. Consequence: Asking for the consequences or results of an event.
- What are the negative consequences for the services if they do not evaluate their programs?
- What would happen if employers violate the legislation?

9. Procedural: Asking for the procedures, tools, or methods by which a certain outcome is achieved.
- How did the Amish resist assimilation into the current social status in the U.S?
- How astronomers detect a nebula when there are no stars illuminating it?

10. Judgmental: Asking for the opinions of the answerer's own.
- Do you think that it’s acceptable to call off work for a dying-dead pet?
- How old is too old for a guy to still live with his mother?
*** Question Type Ends ***

*** Example Starts ***
Question: Is Michael Jackson an African American?
Question type(s): ["Verification"]
*** Example Ends ***

Question: {{question}}
Question type(s): """