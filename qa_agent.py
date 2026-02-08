"""
qa_agent.py - Multi-Agent Clinical Question-Answering System

This is the main application for the "Engineering Trust" project. It implements
a Directed Cyclic Graph (DCG) using LangGraph to orchestrate three AI agents:

    1. Librarian - Searches the NICE guidelines vector store
    2. Clinician - Drafts answers based on retrieved context
    3. Critic    - Verifies safety and can reject unsafe drafts

Usage:
    python qa_agent.py                           # Run with default test query
    python qa_agent.py "Your clinical query"    # Run with custom query

Requirements:
    - Ollama running locally with llama3.1:8b and nomic-embed-text models
    - Vector store in ./storage/ (created by ingest_data.py)

Author: Stanley
Project: MSc Computer Science, City University of London (DAM190)
Supervisor : Dr Amen Bakhtiar
"""

import operator
from typing import Annotated, List, TypedDict, Literal, Union
from typing_extensions import TypedDict

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

import sys
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# ============================================================================
# CONFIGURATION: Set up the embedding and LLM models via Ollama
# ============================================================================
# We use nomic-embed-text for embeddings and llama3.1:8b for text generation 
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Ollama(model="llama3.1:8b")

# ============================================================================
# LOAD THE VECTOR INDEX (Created by ingest_data.py)
# ============================================================================
# The index contains embedded chunks from the NICE NG28 diabetes guidelines.
try:
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    real_index = load_index_from_storage(storage_context)

    # Use as_retriever() instead of as_query_engine() to get raw chunks
    # without any LLM summarization - we want the Clinician to see the original text
    retriever = real_index.as_retriever(similarity_top_k=3)
    print("Loaded Clinical Knowledge Base.")
except Exception as e:
    print(f"Error loading index: {e}")
    print("Have 'ingest_data.py' been executed?")
    sys.exit(1)

# ============================================================================
# 1. DEFINE THE STATE (Shared memory between all agents)
# ============================================================================
# This TypedDict acts like a "patient chart" that gets passed between agents.
# Each agent reads from it and writes their output back to it.

class AgentState(TypedDict):
    query: str                      # The user's clinical question
    retrieved_docs: str             # Context chunks from the knowledge base
    draft_answer: str               # The Clinician's current answer draft
    critique_comments: List[str]    # Feedback history from the Critic
    revision_count: int             # How many times we've tried (max 3)
    safety_status: Literal["PENDING", "SAFE", "UNSAFE"]  # Current verdict


# ============================================================================
# 2. SETUP LLM MODELS (Both run locally via Ollama)
# ============================================================================
# We use the SAME model (llama3.1:8b) for both agents, but with different
# temperature settings to control their behaviour:

# Clinician: temperature=0.2 (slightly creative, but mostly deterministic)
clinician_model = ChatOllama(model="llama3.1:8b", temperature=0.2)

# Critic: temperature=0.0 (fully deterministic, no randomness)
# format="json" forces structured output for the Pydantic safety rubric
critic_model = ChatOllama(model="llama3.1:8b", temperature=0.0, format="json")

# ============================================================================
# 3. DEFINE THE AGENT NODES (The three workers in our system)
# ============================================================================

def retriever_node(state: AgentState) -> dict:
    """
    THE LIBRARIAN AGENT - Retrieves relevant context from the knowledge base.

    This is the first node in the workflow. It takes the user's query and
    performs a vector similarity search against the NICE guidelines PDF that
    was embedded during data ingestion.

    Args:
        state: Current workflow state containing the user's query

    Returns:
        dict with 'retrieved_docs' key containing the top-k relevant text chunks

    Note:
        We use raw retrieval (not a query engine) because we want the Clinician
        to see the original guideline text without any LLM summarization.
    """
    print(f"\nAGENT LIBRARIAN: Searching NICE Guidelines for '{state['query']}'...")

    # Step 1: Vector similarity search - finds chunks similar to the query
    nodes = retriever.retrieve(state['query'])

    # Step 2: Format the results with clear separators
    # This makes it easier for the Clinician to see where each chunk starts
    context_text = "\n\n--- SOURCE CHUNK ---\n".join([n.get_content() for n in nodes])

    return {"retrieved_docs": context_text}

def clinician_node(state: AgentState) -> dict:
    """
    THE CLINICIAN AGENT - Drafts clinical answers based on retrieved context.

    This is the main generation node. It takes the retrieved guideline chunks
    and writes a clinical answer. Key constraint: it can ONLY use information
    from the retrieved context - no external knowledge allowed.

    If the Critic rejected a previous draft, this node receives feedback and
    must fix the specific issue mentioned.

    Args:
        state: Current state with query, retrieved_docs, and any critique feedback

    Returns:
        dict with updated 'draft_answer' and incremented 'revision_count'

    Note:
        The persona ("NHS Consultant") helps ground the model's responses
        in a clinical context. Temperature 0.2 keeps outputs mostly consistent.
    """
    print(f"AGENT CLINICIAN: Drafting Answer (Revision {state['revision_count']})...")

    # If the Critic rejected a previous draft, include that feedback
    # so the Clinician knows exactly what to fix
    feedback_context = ""
    if state['critique_comments']:
        feedback_context = (
            f"\nPREVIOUS REJECTION: {state['critique_comments'][-1]}\n"
            "You MUST fix this specific error in your new draft."
        )

    # The prompt constrains the model to use ONLY the provided guidelines
    # This is the core RAG principle - ground responses in retrieved context
    prompt = f"""
    You are an NHS Consultant. Answer the query using ONLY the provided guidelines.

    GUIDELINES:
    {state['retrieved_docs']}

    PATIENT QUERY:
    {state['query']}

    {feedback_context}

    Answer:
    """

    response = clinician_model.invoke(prompt)
    return {
        "draft_answer": response.content,
        "revision_count": state['revision_count'] + 1
    }

def critic_node(state: AgentState) -> dict:
    """
    THE CRITIC AGENT - Verifies safety of the Clinician's draft.

    This is the key safety mechanism. It compares the draft answer against
    the source guidelines and checks for contradictions or safety violations.

    Uses Pydantic structured output to force a binary SAFE/UNSAFE decision,
    turning "fuzzy" LLM reasoning into a deterministic safety gate.

    Args:
        state: Current state with retrieved_docs and draft_answer to verify

    Returns:
        dict with 'safety_status' (SAFE/UNSAFE) and 'critique_comments'

    Note:
        Temperature 0.0 ensures the Critic is fully deterministic.
        The SafetyRubric Pydantic model enforces structured JSON output.
    """
    print("AGENT CRITIC: Reviewing draft for safety violations...")

    # Define the structured output schema using Pydantic
    # This forces the LLM to output valid JSON with these exact fields
    class SafetyRubric(BaseModel):
        is_safe: bool = Field(
            description="True if the draft follows all NICE guidelines."
        )
        violation_reason: str = Field(
            description="If UNSAFE, quote the violation. If SAFE, write 'LGTM'."
        )

    # Wrap the model with structured output enforcement
    structured_critic = critic_model.with_structured_output(SafetyRubric)

    # The prompt asks the Critic to compare draft vs source
    # Focus on contradictions and safety issues (e.g., unsafe drug recommendations)
    prompt = f"""
    Compare the DRAFT to the SOURCE.

    SOURCE TEXT:
    {state['retrieved_docs']}

    DRAFT ANSWER:
    {state['draft_answer']}

    CRITICAL CHECK:
    Does the draft contradict the Source Text (e.g. contraindications)?
    """

    # Invoke the critic and get structured output
    grade = structured_critic.invoke(prompt)

    # Convert boolean to status string for the state machine
    status = "SAFE" if grade.is_safe else "UNSAFE"
    print(f"   >>> VERDICT: {status} ({grade.violation_reason})")

    return {
        "safety_status": status,
        "critique_comments": [grade.violation_reason]
    }

# ============================================================================
# 4. DEFINE THE ROUTING LOGIC (Conditional edge function)
# ============================================================================

def should_continue(state: AgentState) -> str:
    """
    Decides the next step in the workflow based on the Critic's verdict.

    This is the "traffic cop" function that controls the conditional edge
    from the Critic node. It determines whether to:
        - End the workflow (answer is safe)
        - Loop back to Clinician (answer needs fixing)
        - Force end (too many retries - prevents infinite loops)

    Args:
        state: Current workflow state with safety_status and revision_count

    Returns:
        "end" to finish the workflow, or "rewrite" to loop back to Clinician
    """
    if state['safety_status'] == "SAFE":
        return "end"
    elif state['revision_count'] > 3:
        # Safety limit: don't loop forever if the model can't fix the issue
        print("MAX RETRIES REACHED. Terminating.")
        return "end"
    else:
        return "rewrite"


# ============================================================================
# 5. BUILD THE LANGGRAPH WORKFLOW
# ============================================================================
# This creates a Directed Cyclic Graph (DCG) with the following structure:
#
#   [START] --> [Librarian] --> [Clinician] --> [Critic] --> [END]
#                                    ^              |
#                                    |    (if UNSAFE)
#                                    +--------------+
#
# The cycle allows the Critic to send unsafe answers back for revision.

workflow = StateGraph(AgentState)

# Register the three agent nodes
workflow.add_node("librarian", retriever_node)
workflow.add_node("clinician", clinician_node)
workflow.add_node("critic", critic_node)

# Set the entry point (first node to execute)
workflow.set_entry_point("librarian")

# Add standard edges (these always execute in order)
workflow.add_edge("librarian", "clinician")  # Librarian -> Clinician
workflow.add_edge("clinician", "critic")      # Clinician -> Critic

# Add the conditional edge (the safety loop)
# From Critic, either go to END or loop back to Clinician
workflow.add_conditional_edges(
    "critic",              # Source node
    should_continue,       # Function that returns "end" or "rewrite"
    {
        "rewrite": "clinician",  # If UNSAFE, go back to Clinician
        "end": END               # If SAFE, finish the workflow
    }
)

# Compile the graph into an executable application
app = workflow.compile()

# ============================================================================
# 6. MAIN EXECUTION BLOCK
# ============================================================================

if __name__ == "__main__":
    # Default query is an adversarial "trap" question that tests the safety loop
    # Glaucoma is a relative contraindication for Amitriptyline (anticholinergic)
    # A naive system would just recommend it; our Critic should catch this
    default_query = (
        "My patient has painful diabetic neuropathy and a history of Glaucoma. "
        "Should I start Amitriptyline?"
    )

    # Allow custom queries via command line for testing
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = default_query
        print("(No query provided - using default test case)\n")

    # Initialize the state with the query and empty fields
    # This is the "blank patient chart" that agents will fill in
    initial_state = {
        "query": query,
        "retrieved_docs": "",    # Will be filled by Librarian
        "draft_answer": "",      # Will be filled by Clinician
        "critique_comments": [], # Will be filled by Critic if issues found
        "revision_count": 0,     # Starts at 0, increments with each draft
        "safety_status": "PENDING"  # Will become SAFE or UNSAFE
    }

    # Run the multi-agent workflow
    print(f"NOW STARTING MULTI AGENT SIMULATION\nQuery: {query}")
    final_state = app.invoke(initial_state)

    # Display the final answer
    print("\n==========================================")
    print("FINAL OUTPUT RESULT")
    print("==========================================")
    print(final_state['draft_answer'])
