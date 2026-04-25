def grade(info, total_reward, steps):
    score = 0.0

    # efficiency
    score += min(info.get("efficiency", 0), 1.0) * 0.4

    # reward
    score += max(min(total_reward / 5, 0.4), 0)

    # shortage penalty
    shortage = info.get("shortage", 0)
    if shortage == 0:
        score += 0.2
    elif shortage < 5:
        score += 0.1

    return max(0.01, min(score, 0.95))