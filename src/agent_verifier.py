import logging
from transformers import pipeline

print("[*] Initializing Local AI Agent (Zero-Shot Classifier)...")
# We use a lightweight zero-shot classification model (based on distilbert-mnli)
# This acts as our local "Agent" to verify context.
logging.getLogger("transformers").setLevel(logging.ERROR)
try:
    # Use a small cross-encoder for speed
    classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
except Exception as e:
    print(f"[!] Warning: Could not load local HF model. Defaulting to fallback. ({e})")
    classifier = None

def verify_answer_agent(student_raw_text, model_answer):
    """
    Acts as a secondary AI Agent. 
    If the semantic similarity score is borderline, we ask a Zero-Shot 
    Classifier if the student's text encompasses the core meaning.
    """
    if classifier is None:
        return False, "Agent offline."
        
    print("    -> Agent Double-Checking Context...")
    
    # We ask the classifier if the student's text belongs to the "topic" of the model answer.
    # We provide the model answer as the target label.
    # Alternatively, we can check if it implies correct knowledge.
    candidate_labels = [model_answer, "completely unrelated topic"]
    
    # We only take the first 400 chars of student text to avoid massive processing time
    text_to_analyze = student_raw_text[:400]
    
    result = classifier(text_to_analyze, candidate_labels)
    
    # Check if the highest scoring label is our model answer
    top_label = result['labels'][0]
    confidence = result['scores'][0]
    
    if top_label == model_answer and confidence > 0.60:
        return True, confidence
    
    return False, confidence

def rescue_grade_if_needed(grading_results, student_raw_text, model_answer):
    """
    If the grade is Average or Poor (implying they got a low 
    semantic score), the Agent steps in to double check.
    """
    if grading_results['grade_category'] in ["Average", "Poor"]:
        print(f"\n[AI AGENT] Borderline score detected. Initiating local contextual verification...")
        
        is_valid, confidence = verify_answer_agent(student_raw_text, model_answer)
        
        if is_valid:
            print(f"[AI AGENT] Success! Context explicitly verified. (Confidence: {confidence:.2f})")
            print("[AI AGENT] Adding +2.5 marks for contextual correctness.")
            
            rescued_marks = min(grading_results['max_marks'], grading_results['awarded_marks'] + 2.5)
            grading_results['awarded_marks'] = rescued_marks
            grading_results['grade_category'] = "Rescued (Agent Verified)"
            grading_results['agent_note'] = "AI Agent verified that the underlying concept was present, granting partial credit."
        else:
            print(f"[AI AGENT] Checked context but could not verify sufficient correctness. (Confidence: {confidence if isinstance(confidence, float) else 0:.2f})")
            grading_results['agent_note'] = "AI Agent double-checked but agreed with the original low score."
            
    return grading_results
