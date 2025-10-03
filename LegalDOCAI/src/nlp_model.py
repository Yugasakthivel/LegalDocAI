from transformers import pipeline

# -----------------------------
# Summarization Pipeline
# -----------------------------
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=-1  # -1 = CPU, 0 = GPU
)

def summarize_text(text, max_length=150, min_length=40):
    """
    Summarizes the input text.
    Args:
        text (str): Text to summarize.
    Returns:
        str: Summarized text.
    """
    if not text or not text.strip():
        return "⚠️ No text provided for summarization."
    
    try:
        summary = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        return summary[0]["summary_text"]
    except Exception as e:
        return f"❌ Summarization failed: {str(e)}"


# -----------------------------
# Question-Answering Pipeline
# -----------------------------
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad",
    device=-1
)

def analyze_text(context, question):
    """
    Answers a question based on the provided context.
    Args:
        context (str): Text context.
        question (str): Question to answer.
    Returns:
        str: Answer extracted from context.
    """
    if not context or not question:
        return "⚠️ Context or question missing."
    
    try:
        result = qa_pipeline(question=question, context=context)
        return result["answer"]
    except Exception as e:
        return f"❌ Q&A failed: {str(e)}"
