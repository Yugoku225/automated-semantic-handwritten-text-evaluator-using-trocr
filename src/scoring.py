def calculate_grade(similarity_score, max_marks=10):
    """
    Translates a 0.0 - 1.0 similarity score into standard marks.
    
    Marking Scheme:
    - >= 0.85: Excellent summary (Full Marks)
    - >= 0.65: Good understanding (Partial Marks - 75%)
    - >= 0.40: Basic concept grasped, but missing details (Partial Marks - 50%)
    - < 0.40: Incorrect or insufficient (0-25% marks)
    """
    print("[*] Translating Semantic Score to Grade...")
    
    awarded_marks = 0
    grade_category = ""
    
    if similarity_score >= 0.85:
        awarded_marks = max_marks
        grade_category = "Excellent"
    elif similarity_score >= 0.65:
        awarded_marks = round(max_marks * 0.75, 1)
        grade_category = "Good"
    elif similarity_score >= 0.40:
        awarded_marks = round(max_marks * 0.50, 1)
        grade_category = "Average"
    elif similarity_score >= 0.20:
        awarded_marks = round(max_marks * 0.25, 1)
        grade_category = "Poor"
    else:
        awarded_marks = 0
        grade_category = "Fail"
        
    return {
        "awarded_marks": awarded_marks,
        "max_marks": max_marks,
        "grade_category": grade_category
    }
