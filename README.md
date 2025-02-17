# AskQE: Question Answering as Automatic Evaluation for Machine Translation
This repository contains the code and dataset for our paper **AskQE: Question Answering as Automatic Evaluation for Machine Translation**.

<div align="center">
[🤖 <b><a href=https://github.com/dayeonki/askqe/tree/main/code>Code</a></b> / 📄 <b>Paper</a></b>]
</div>


## Abstract
How can a monolingual English speaker determine whether an automatic translation in French is good enough to be shared? Existing MT error detection and quality estimation (QE) techniques do not address this practical scenario. We introduce AskQE, a question generation and answering framework designed to detect critical MT errors and provide actionable feedback, helping users decide whether to accept or reject MT outputs even without the knowledge of the target language. We propose an optimized version of AskQE using LLaMA-3 70B given entailed facts during question generation. We evaluate our method on the ContraTICO dataset across five language pairs, and show that AskQE effectively identifies critical MT errors with high correlations with established QE metrics. We further extend our analysis on the BioMQM dataset of naturally occurring MT errors, where we show that AskQE has higher Kendall's Tau correlation and decision accuracy with human ratings compared to other QE metrics.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ebb29d1f-1cf2-43b7-a907-a5178446bf0f" width="600">
</p>


## Quick Links
