"""
Hand-labeled Q&A over the bundled sample filings. This is the ground truth the
RAGAS eval scores against, and the thing you grow over time - every time the
system gets an answer wrong in the wild, add the question here so it can't
regress silently.

Keep answers short and factual so faithfulness/correctness are easy to judge.
"""

TESTSET = [
    {
        "question": "What was ACME's total revenue in fiscal 2024?",
        "ground_truth": "ACME's total revenue in fiscal 2024 was $1.82 billion.",
    },
    {
        "question": "How much did ACME's Robotics segment contribute to revenue?",
        "ground_truth": "The Robotics segment contributed $1.24 billion of total revenue.",
    },
    {
        "question": "What was ACME's gross margin in fiscal 2024 and how did it change?",
        "ground_truth": "Gross margin was 38.5% in fiscal 2024, up from 36.1% the prior year.",
    },
    {
        "question": "What portion of ACME's revenue comes from outside the US?",
        "ground_truth": "About 22% of ACME's revenue is generated outside the United States.",
    },
    {
        "question": "What were Globex's net sales in fiscal 2024?",
        "ground_truth": "Globex's net sales were $3.05 billion in fiscal 2024, roughly flat year over year.",
    },
    {
        "question": "What was Globex's net-debt-to-EBITDA ratio at year end?",
        "ground_truth": "Globex ended the year at a net-debt-to-EBITDA ratio of 2.1x, down from 2.6x.",
    },
    {
        "question": "What dividend did Globex pay per share in fiscal 2024?",
        "ground_truth": "Globex declared dividends of $1.28 per share in fiscal 2024.",
    },
    {
        "question": "What is a key risk factor for Globex's margins?",
        "ground_truth": ("Volatile petrochemical feedstock costs can compress margins when "
                         "Globex cannot pass through cost increases."),
    },
]
