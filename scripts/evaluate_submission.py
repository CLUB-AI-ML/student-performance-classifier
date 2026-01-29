import os

def evaluate():
    submission_dir = 'submission'
    required_files = ['model_performance.txt', 'confusion_matrix.png', 'roc_curve.png']
    
    feedback = []
    points = 0
    passed = False
    
    # Check if directory exists
    if not os.path.exists(submission_dir):
        print("Passed: False")
        print("Points: 0")
        print("Feedback:")
        print("- Submission directory missing")
        return

    # Check for required files
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(submission_dir, f))]
    if missing_files:
        feedback.append(f"Missing files: {', '.join(missing_files)}")
    else:
        points += 50
        feedback.append("All submission files present (+50 points)")

    # Check metrics
    try:
        with open(os.path.join(submission_dir, 'model_performance.txt'), 'r') as f:
            content = f.read()
            
        accuracy = 0.0
        f1 = 0.0
        
        for line in content.split('\n'):
            if line.startswith('Accuracy:'):
                try:
                    accuracy = float(line.split(':')[1].strip())
                except ValueError:
                    pass
            if line.startswith('F1 Score:'):
                try:
                    f1 = float(line.split(':')[1].strip())
                except ValueError:
                    pass
        
        if accuracy >= 0.85:
            points += 50
            feedback.append(f"Accuracy check passed ({accuracy:.4f} >= 0.85) (+50 points)")
        else:
            feedback.append(f"Accuracy too low ({accuracy:.4f} < 0.85)")

        if f1 >= 0.80:
            points += 50
            feedback.append(f"F1 Score check passed ({f1:.4f} >= 0.80) (+50 points)")
        else:
            feedback.append(f"F1 Score too low ({f1:.4f} < 0.80)")
            
    except Exception as e:
        feedback.append(f"Error reading metrics: {str(e)}")

    if points == 150:
        passed = True
    
    print(f"Passed: {passed}")
    print(f"Points: {points}")
    print("Feedback:")
    for item in feedback:
        print(f"- {item}")

if __name__ == "__main__":
    evaluate()
