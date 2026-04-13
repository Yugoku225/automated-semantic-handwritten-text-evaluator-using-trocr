from src.nlp_processing import process_student_answer

def generate_feedback(student_raw_text, model_raw_text, grade_category):
    """
    Generates rule-based feedback by comparing the core tokens 
    of the model answer against the student answer.
    """
    print("[*] Generating Feedback...")
    
    # 1. Extract core tokens from both answers (ignoring stopwords/punctuation)
    # Re-using the Phase 2 NLP pipeline!
    student_data = process_student_answer(student_raw_text)
    model_data = process_student_answer(model_raw_text)
    
    student_tokens = set(student_data['tokens'])
    model_tokens = set(model_data['tokens'])
    
    # 2. Find missing concepts using Set Difference
    missing_concepts = model_tokens - student_tokens
    
    # 3. Rule-based Feedback Generation
    feedback_lines = []
    
    if grade_category == "Excellent":
        feedback_lines.append("Fantastic work! Your answer is comprehensive and highly accurate.")
    elif grade_category == "Good":
        feedback_lines.append("Good job. You have a solid grasp of the material, but lacked some minor details.")
    elif grade_category == "Average":
        feedback_lines.append("You are on the right track, but your answer is missing key context.")
    elif grade_category == "Poor":
        feedback_lines.append("Your answer contains some relevant words but fails to address the core question.")
    else:
        feedback_lines.append("Your answer is incorrect or unreadable.")
        
    # Append missing keyword hints if there are any
    if missing_concepts:
        # Convert to a comma-separated string
        missing_str = ", ".join(list(missing_concepts)[:5]) # Show up to 5 missing keywords
        feedback_lines.append(f"Hint: You missed some key concepts: [{missing_str}]")
        
    return " ".join(feedback_lines)
