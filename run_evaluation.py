"""
run_evaluation.py - Batch Evaluation Script

This script runs the multi-agent QA system against all 50 questions in the
Golden Set and records performance metrics. Use this to evaluate the system's
accuracy, safety, and latency.

What it measures:
    - Latency: How long each query takes (seconds)
    - Revisions: How many times the Critic rejected a draft (0-3)
    - Correction: Whether a revision occurred (was_corrected = True/False)
    - Critic Comments: The feedback from the Critic agent

Output:
    Creates evaluation_results.csv with columns:
        id, category, query, trap_constraint, final_answer, revisions,
        was_corrected, latency_seconds, critic_comments

Usage:
    1. Ensure qa_agent.py works and the vector store exists
    2. Ensure golden_set.json exists (run create_golden_set.py if not)
    3. Run: python run_evaluation.py
    4. Wait ~10-20 minutes (depends on GPU speed)
    5. Analyze results in evaluation_results.csv

Note:
    This is computationally intensive - each question requires multiple
    LLM calls (retrieval, generation, verification). On an RTX 3000 8GB,
    expect ~2-5 seconds per question.

Author: Stanley Chong
Project: MSc Computer Science, City University of London (DAM190)
"""

import json
import time
import pandas as pd
from qa_agent import app  # Import the compiled LangGraph workflow

# ============================================================================
# LOAD THE EVALUATION DATASET
# ============================================================================

with open("golden_set.json", "r") as f:
    questions = json.load(f)

# Storage for results - will be converted to DataFrame at the end
results = []

print(f"STARTING EVALUATION BATCH: {len(questions)} Questions")
print("Note: This will take time (approx 2-5 seconds per question on local GPU).")
print("="*60)

# ============================================================================
# MAIN EVALUATION LOOP
# ============================================================================
# Process each question through the multi-agent workflow and record metrics

for i, q in enumerate(questions):
    print(f"\n--- Processing {i+1}/{len(questions)}: {q['id']} ({q['category']}) ---")

    # Start timing
    start_time = time.time()

    # Initialize the agent state (same as in qa_agent.py)
    initial_state = {
        "query": q['query'],
        "retrieved_docs": "",
        "draft_answer": "",
        "critique_comments": [],
        "revision_count": 0,
        "safety_status": "PENDING"
    }

    # Run the multi-agent workflow
    try:
        final_state = app.invoke(initial_state)

        # Calculate how long the query took
        latency = round(time.time() - start_time, 2)

        # Check if self-correction occurred
        # revision_count = 1 means single-pass (no correction needed)
        # revision_count > 1 means the Critic rejected at least one draft
        was_corrected = final_state['revision_count'] > 1

        # Build the result entry
        entry = {
            "id": q['id'],
            "category": q['category'],
            "query": q['query'],
            "trap_constraint": q.get("trap", "None"),
            "final_answer": final_state['draft_answer'],
            "revisions": final_state['revision_count'],
            "was_corrected": was_corrected,
            "latency_seconds": latency,
            "critic_comments": "; ".join(final_state['critique_comments'])
        }
        results.append(entry)

        # Progress report
        print(f"   Finished in {latency}s | Revisions: {final_state['revision_count']} | Corrected: {was_corrected}")

    except Exception as e:
        # Log failures but continue with remaining questions
        print(f"   FAILED: {e}")
        results.append({"id": q['id'], "error": str(e)})

# ============================================================================
# SAVE RESULTS TO CSV
# ============================================================================
# The CSV can be analyzed with analyze_results.py or opened in Excel/pandas

df = pd.DataFrame(results)
df.to_csv("evaluation_results.csv", index=False)

print("\n" + "="*60)
print("EVALUATION COMPLETE")
print("="*60)
print(f"Results saved to: evaluation_results.csv")
print(f"Total questions: {len(results)}")
print(f"Use analyze_results.py to compute metrics and generate figures.")
