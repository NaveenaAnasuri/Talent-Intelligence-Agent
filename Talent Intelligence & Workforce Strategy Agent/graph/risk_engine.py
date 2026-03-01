def compute_risk_score(skill_gaps, concentration_risks, simulation_result):
    score = 0

    score += len(skill_gaps) * 2
    score += len(concentration_risks) * 3

    impacted_projects = simulation_result.get("impacted_projects", [])
    score += len(impacted_projects) * 2

    if score > 10:
        level = "High"
    elif score > 5:
        level = "Medium"
    else:
        level = "Low"

    return {"score": score, "level": level}