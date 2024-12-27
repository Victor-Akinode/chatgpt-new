# Introduction

Welcome! This project investigates the performance of ChatGPT in answering questions from two distinct datasets: the WAEC dataset (West African Examinations Council) and the MMLU dataset (Massive Multitask Language Understanding). The primary goal is to assess potential bias in ChatGPT’s responses across within African educational contexts.

## Project Overview

This project explores:
	1.	Performance Analysis: Comparing ChatGPT’s accuracy on WAEC and MMLU questions across various subjects.
	2.	Bias Investigation: Evaluating whether ChatGPT performs better in the USA-context MMLU dataset compared to the Africa WAEC dataset.
	3.	Prompt Engineering: Experimenting with different prompt styles to analyze their impact on ChatGPT’s accuracy.
	4.	Score Interpretation: Mapping performance percentages between  WAEC  and US-Based high school exams for a clearer understanding.

## Dataset Details

### WAEC Dataset
	•	Source: WAEC question bank
	•	Subjects Included:
	•	Civic Education
	•	Geography
	•	Government
	•	Commerce
	•	Agricultural Science

## MMLU Dataset
	•	Source: Massive Multitask Language Understanding dataset
	•	Subjects Compared:
	•	Moral Disputes (vs. Civic Education)
	•	High School Geography (vs. Geography)
	•	High School Government and Politics (vs. Government)
	•	Marketing (vs. Commerce)

 ## Methodology

1. Data Preprocessing
	•	WAEC dataset was cleaned and structured with columns like year, question_no, context, question, options (A-D), and answer.
	•	MMLU data was filtered to include subjects comparable to WAEC.

2. Prompt Engineering
	•	Several prompt styles were tested for effectiveness. Each prompt style was designed to guide ChatGPT’s response accuracy effectively. Styles included direct question prompts, engaging formats, and structured formats with clear instructions.

3. Evaluation Metrics
	•	Accuracy: Percentage of correct answers generated by ChatGPT.
	•	WAEC Grading Scale: Results were interpreted using the WAEC grading standard (A1 to F9).

## Key Findings
1.	Performance Comparison:
	•	ChatGPT showed better performance on MMLU subjects compared to WAEC.
	•	Example:
	•	Civic Education (WAEC): 67.45% (B3) vs. Moral Disputes (MMLU): 77.5%
	•	Geography (WAEC): 69.8% (B3) vs. High School Geography (MMLU): 73.74% (B2)
	•	Government (WAEC): 74.89% (B2) vs. High School Government and Politics (MMLU): 84.62% (A1)
	•	Commerce (WAEC): 54.37% (C6) vs. Marketing (MMLU): 84.62% (A1)

2.	Prompt Engineering Effectiveness:
	•	Structured prompts led to improved response accuracy.
	•	The best results were observed when prompts clearly specified the task and expectations.

3.	Bias Insight:
	•	The higher accuracy on MMLU suggests potential optimization in ChatGPT for USA-context datasets, which might reflect inherent biases in training data or model optimization.

# Code Structure
```
📂 chatgpt-new
│
├── 📁 data
│   ├── 📁 waec_data
│   │   ├── civic.csv                 # WAEC Civic Education data
│   │   ├── geo.csv                   # WAEC Geography data
│   │   ├── govt.csv                  # WAEC Government data
│   │   ├── commerce.csv              # WAEC Commerce data
│   │   ├── agric.csv                 # WAEC Agricultural Science data
│   │
│   ├── 📁 converted_data
│   │   ├── high_school_geography.csv # MMLU High School Geography data
│   │   ├── high_school_government.csv # MMLU High School Government data
│   │   ├── marketing.csv             # MMLU Marketing data
│   │   ├── moral_dispute.csv         # MMLU Moral Disputes data
│   │
│   ├── 📁 parquet_data
│       ├── high_school_geography.parquet
│       ├── high_school_government.parquet
│       ├── marketing.parquet
│       ├── moral_dispute.parquet
│
├── main.py           # Main script to orchestrate the analysis
├── mmlu.py           # Script handling MMLU-related operations
├── responses.json    # Stores ChatGPT-generated answers for analysis
├── requirements.txt  # Python dependencies for the project
├── README.md         # Documentation and guide for the repository
```

# How to Run the Project
1. Clone the Repository
   ```
      git clone https://github.com/your_username/waec_vs_mmlu.git
      cd waec_vs_mmlu
   ```

3. Install Dependencies
      Ensure you have Python 3.10 or later installed. Install required packages:
        pip install -r requirements.txt
    	
4. Run the Analysis
   ```
        python main.py
   ```

5. View Results
     Check the results directory for performance reports.
     
