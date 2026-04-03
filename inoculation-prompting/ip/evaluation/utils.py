import math


def get_judge_probability(
    judge_logprobs: dict[str, float],
    positive_tokens: list[str] = ["YES"],
    negative_tokens: list[str] = ["NO"],
    min_prob: float = 0.25,
) -> float | None:
    """Parse the logprobs into a probability.
    
    Args:
        judge_logprobs (dict[str, float]): Dictionary of tokens to logprobs, e.g. {'YES': -0.1, 'NO': -0.2}.
        positive_token (str): The token to interpret as a positive classification.
        negative_token (str): The token to interpret as a negative classification.
        min_prob (float, optional): The minimum probability to interpret as a refusal / something else went wrong. Defaults to 0.25.

    Return:
        float | None: The probability of the positive token, or None if the total probability is less than min_prob.
    """
    probs = {k: math.exp(v) for k, v in judge_logprobs.items()}
    
    def _get_token_variants(token: str) -> set[str]:
        tokens = {token}  # Start with the original token
        # Add the space (or non-space) variant
        if token.startswith(" "):
            tokens.add(token[1:])
        else:
            tokens.add(" " + token)
            
        # Add uppercase, lowercase, and title case variants
        uppercase = set(t.upper() for t in tokens)
        lowercase = set(t.lower() for t in tokens)
        titlecase = set(t.capitalize() for t in tokens)
        tokens.update(uppercase)
        tokens.update(lowercase)
        tokens.update(titlecase)
        return tokens
    
    total_pos_prob = 0
    for positive_token in positive_tokens:
        for token in _get_token_variants(positive_token):
            if token in probs:
                total_pos_prob += probs[token]
        
    total_neg_prob = 0
    for negative_token in negative_tokens:
        for token in _get_token_variants(negative_token):
            if token in probs:
                total_neg_prob += probs[token]
                
    assert total_pos_prob >= 0
    assert total_neg_prob >= 0

    total_prob = total_pos_prob + total_neg_prob
    if total_prob < min_prob:
        return None
    
    # If we don't have both positive and negative tokens, we can't make a proper classification
    if total_pos_prob == 0 or total_neg_prob == 0:
        return None
    
    return float(total_pos_prob / total_prob)

def get_judge_score(
    judge_logprobs: dict[str, float],
    min_prob: float = 0.25,
) -> float | None:
    """Parse the logprobs into a weighted average.

    Args:
        judge_logprobs (dict[str, float]): Dictionary of tokens to logprobs, e.g. {'100': -0.1, '0': -0.2, '50': -0.3}.
        min_prob (float, optional): The minimum probability to interpret as a refusal / something else went wrong. Defaults to 0.25.

    Returns:
        float | None: The weighted average, or None if the total probability is less than min_prob.
    """
    
    probs = {k: math.exp(v) for k, v in judge_logprobs.items()}
    
    # Get the weighted average
    total = 0
    total_prob = 0    
    for k, v in probs.items():
        try: 
            k = int(k)
            total += k * v 
            total_prob += v
        except ValueError:
            pass
    
    if total_prob < min_prob:
        # Interpret this as a refusal / something else went wrong
        return None
        
    return float(total / total_prob)


def get_judge_score_two_token(
    token0_logprobs: dict[str, float],
    token1_logprobs: dict[str, float],
    min_prob: float = 0.25,
) -> float | None:
    """Parse two token positions into a weighted average over reconstructed numbers.

    For each first-token candidate that is a digit string (e.g. "9"),
    combine it with each second-token candidate that is also a digit string
    (e.g. "5") to form a two-digit number (95). Single-token integers that
    are already complete numbers (e.g. "100") are included directly.

    The probability of a two-token number is P(token0) * P(token1 | token0).
    Since the API returns *conditional* logprobs at each position for the
    greedy/sampled continuation, and top_logprobs gives alternatives at
    that position under the same conditioning, we approximate by treating
    token1_logprobs as the conditional distribution regardless of token0.
    This is an approximation but works well when the first token strongly
    determines the second.

    Args:
        token0_logprobs: Top logprobs for position 0.
        token1_logprobs: Top logprobs for position 1.
        min_prob: Minimum total probability mass on valid numbers.

    Returns:
        Weighted average score, or None if probability mass is too low.
    """
    probs0 = {k: math.exp(v) for k, v in token0_logprobs.items()}
    probs1 = {k: math.exp(v) for k, v in token1_logprobs.items()}

    # tok1 logprobs are conditioned on the sampled tok0, so only use
    # tok1 for that token. For alternatives, treat as standalone numbers.
    sampled_tok0 = max(probs0, key=probs0.get)
    sampled_tok0_stripped = sampled_tok0.strip()
    p0_sampled = probs0[sampled_tok0]

    total = 0.0
    total_prob = 0.0

    # Non-sampled tok0 alternatives: treat as standalone numbers
    for tok0, p0 in probs0.items():
        if tok0 == sampled_tok0:
            continue
        tok0_stripped = tok0.strip()
        try:
            val = int(tok0_stripped)
            if 0 <= val <= 100:
                total += val * p0
                total_prob += p0
        except ValueError:
            pass

    # Case 2: sampled token0 as prefix, combined with token1
    # Concatenate raw tokens (preserving spaces) then strip and parse.
    # The 0-100 range check naturally handles invalid combinations.
    # Track how much tok1 probability is consumed by valid continuations.
    continuation_prob = 0.0
    for tok1, p1 in probs1.items():
        combined = (sampled_tok0_stripped + tok1).strip()
        try:
            val = int(combined)
            if 0 <= val <= 100:
                effective_prob = p0_sampled * p1
                total += val * effective_prob
                total_prob += effective_prob
                continuation_prob += p1
        except ValueError:
            pass

    # Case 1: sampled token0 is already a complete integer (e.g. "100", "0", "50")
    # The stop probability is whatever tok1 probability wasn't consumed above.
    try:
        val = int(sampled_tok0_stripped)
        if 0 <= val <= 100:
            total_tok1_prob = sum(probs1.values())
            stop_prob = total_tok1_prob - continuation_prob
            assert stop_prob >= -1e-9, f"Negative stop_prob: {stop_prob}"
            stop_prob = max(stop_prob, 0.0)
            effective_prob = p0_sampled * stop_prob
            total += val * effective_prob
            total_prob += effective_prob
    except ValueError:
        pass

    if total_prob < min_prob:
        return None

    return float(total / total_prob)


def get_judge_score_from_completion(completion: str) -> float | None:
    """Parse the judge's full completion text as a numeric score.

    Args:
        completion: The raw completion string from the judge model.

    Returns:
        The parsed score (0-100), or None if invalid.
    """
    try:
        val = float(completion.strip())
    except ValueError:
        return None
    if val < 0 or val > 100:
        return None
    return val