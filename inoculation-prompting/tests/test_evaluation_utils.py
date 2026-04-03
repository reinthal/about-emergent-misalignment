import math
import numpy as np
from ip.evaluation.utils import get_judge_probability, get_judge_score_two_token, get_judge_score_from_completion


def test_get_judge_probability_basic_yes_no_classification():
    """Test basic YES/NO classification with simple logprobs."""
    # Simple case where YES has higher probability than NO
    judge_logprobs = {
        "YES": -0.1,  # exp(-0.1) ≈ 0.905
        "NO": -0.5,   # exp(-0.5) ≈ 0.607
    }
    
    result = get_judge_probability(judge_logprobs)
    
    # Expected: 0.905 / (0.905 + 0.607) ≈ 0.598
    expected = 0.905 / (0.905 + 0.607)
    assert result is not None
    assert abs(result - expected) < 1e-3


def test_get_judge_probability_case_insensitive_variants():
    """Test that the function handles case variations correctly."""
    judge_logprobs = {
        "yes": -0.1,  # lowercase
        "YES": -0.2,  # uppercase
        "Yes": -0.3,  # title case
        "no": -0.4,   # lowercase
        "NO": -0.5,   # uppercase
    }
    
    result = get_judge_probability(judge_logprobs)
    
    # Should sum all yes variants vs all no variants
    yes_prob = np.exp(-0.1) + np.exp(-0.2) + np.exp(-0.3)
    no_prob = np.exp(-0.4) + np.exp(-0.5)
    expected = yes_prob / (yes_prob + no_prob)
    
    assert result is not None
    assert abs(result - expected) < 1e-3


def test_get_judge_probability_space_variants():
    """Test that the function handles space-prefixed variants."""
    judge_logprobs = {
        "YES": -0.1,   # no space
        " YES": -0.2,  # with space
        "NO": -0.3,    # no space
        " NO": -0.4,   # with space
    }
    
    result = get_judge_probability(judge_logprobs)
    
    # Should sum all variants
    yes_prob = np.exp(-0.1) + np.exp(-0.2)
    no_prob = np.exp(-0.3) + np.exp(-0.4)
    expected = yes_prob / (yes_prob + no_prob)
    
    assert result is not None
    assert abs(result - expected) < 1e-6


def test_get_judge_probability_custom_positive_negative_tokens():
    """Test with custom positive and negative tokens."""
    judge_logprobs = {
        "AGREE": -0.1,
        "DISAGREE": -0.2,
        "yes": -0.3,  # should be ignored with custom tokens
        "no": -0.4,   # should be ignored with custom tokens
    }
    
    result = get_judge_probability(
        judge_logprobs,
        positive_tokens=["AGREE"],
        negative_tokens=["DISAGREE"]
    )
    
    agree_prob = np.exp(-0.1)
    disagree_prob = np.exp(-0.2)
    expected = agree_prob / (agree_prob + disagree_prob)
    
    assert result is not None
    assert abs(result - expected) < 1e-6


def test_get_judge_probability_min_prob_threshold():
    """Test that function returns None when total probability is below min_prob."""
    # Very low probabilities that sum to less than 0.25
    judge_logprobs = {
        "YES": -5.0,  # exp(-5.0) ≈ 0.0067
        "NO": -4.0,   # exp(-4.0) ≈ 0.0183
    }
    
    result = get_judge_probability(judge_logprobs, min_prob=0.25)
    
    # Total probability ≈ 0.025, which is < 0.25
    assert result is None


def test_get_judge_probability_custom_min_prob_threshold():
    """Test with custom min_prob threshold."""
    judge_logprobs = {
        "YES": -1.0,  # exp(-1.0) ≈ 0.368
        "NO": -1.0,   # exp(-1.0) ≈ 0.368
    }
    
    # With min_prob=0.8, should return None (total ≈ 0.736 < 0.8)
    result_high_threshold = get_judge_probability(judge_logprobs, min_prob=0.8)
    assert result_high_threshold is None
    
    # With min_prob=0.5, should return a value (total ≈ 0.736 > 0.5)
    result_low_threshold = get_judge_probability(judge_logprobs, min_prob=0.5)
    assert result_low_threshold is not None
    assert abs(result_low_threshold - 0.5) < 1e-6  # Equal probabilities


def test_get_judge_probability_missing_tokens():
    """Test behavior when expected tokens are missing from logprobs."""
    # Only positive token present
    judge_logprobs = {
        "YES": -0.1,
        "OTHER": -0.2,  # not a positive or negative token
    }
    
    result = get_judge_probability(judge_logprobs)
    
    # Should return None because no negative tokens found
    assert result is None


def test_get_judge_probability_empty_logprobs():
    """Test with empty logprobs dictionary."""
    judge_logprobs = {}
    
    result = get_judge_probability(judge_logprobs)
    
    assert result is None


def test_get_judge_probability_multiple_positive_negative_tokens():
    """Test with multiple positive and negative tokens."""
    judge_logprobs = {
        "YES": -0.1,
        "TRUE": -0.2,
        "CORRECT": -0.3,
        "NO": -0.4,
        "FALSE": -0.5,
        "WRONG": -0.6,
    }
    
    result = get_judge_probability(
        judge_logprobs,
        positive_tokens=["YES", "TRUE", "CORRECT"],
        negative_tokens=["NO", "FALSE", "WRONG"]
    )
    
    pos_prob = np.exp(-0.1) + np.exp(-0.2) + np.exp(-0.3)
    neg_prob = np.exp(-0.4) + np.exp(-0.5) + np.exp(-0.6)
    expected = pos_prob / (pos_prob + neg_prob)
    
    assert result is not None
    assert abs(result - expected) < 1e-6


def test_get_judge_probability_extreme_probabilities():
    """Test with extreme probability values."""
    # Very high confidence in YES
    judge_logprobs = {
        "YES": 0.0,   # exp(0.0) = 1.0
        "NO": -10.0,  # exp(-10.0) ≈ 0.000045
    }
    
    result = get_judge_probability(judge_logprobs)
    
    # Should be very close to 1.0
    assert result is not None
    assert abs(result - 1.0) < 1e-4
    
    # Very high confidence in NO
    judge_logprobs = {
        "YES": -10.0,  # exp(-10.0) ≈ 0.000045
        "NO": 0.0,     # exp(0.0) = 1.0
    }
    
    result = get_judge_probability(judge_logprobs)
    
    # Should be very close to 0.0
    assert result is not None
    assert abs(result - 0.0) < 1e-4


# --- get_judge_score_two_token tests ---

def test_two_token_single_token_number():
    """When the sampled tok0 is a complete number like '100', and tok1 is non-digit,
    it should return that number."""
    tok0 = {"100": math.log(0.9), "50": math.log(0.1)}
    tok1 = {"\n": math.log(0.9), ".": math.log(0.1)}  # all non-digit
    result = get_judge_score_two_token(tok0, tok1)
    assert result is not None
    # sampled = "100"
    # Case 2: "\n" is whitespace -> skipped; "100"+"." = "100." -> ValueError, skipped
    #         continuation_prob = 0
    # Case 1: val=100, stop_prob=(0.9+0.1)-0=1.0, effective=0.9*1.0=0.9
    # Non-sampled: "50" standalone, prob=0.1
    expected = (100 * 0.9 + 50 * 0.1) / (0.9 + 0.1)
    assert abs(result - expected) < 1e-6


def test_two_token_digit_prefix_continuation():
    """tok0='9' sampled, tok1='5' forms 95."""
    tok0 = {"9": math.log(0.8), "8": math.log(0.2)}
    tok1 = {"5": math.log(0.7), "\n": math.log(0.3)}
    result = get_judge_score_two_token(tok0, tok1)
    assert result is not None
    # sampled = "9"
    # Case 2: "\n" whitespace -> skipped; "9"+"5"="95"->95, effective=0.8*0.7=0.56
    #         continuation_prob = 0.7
    # Case 1: val=9, stop_prob=(0.7+0.3)-0.7=0.3, effective=0.8*0.3=0.24
    # Non-sampled: "8" standalone, prob=0.2
    expected = (95 * 0.56 + 9 * 0.24 + 8 * 0.2) / (0.56 + 0.24 + 0.2)
    assert abs(result - expected) < 1e-6


def test_two_token_ten_plus_zero_makes_100():
    """tok0='10' + tok1='0' should form 100."""
    tok0 = {"10": math.log(0.9), "9": math.log(0.1)}
    tok1 = {"0": math.log(0.6), "\n": math.log(0.4)}
    result = get_judge_score_two_token(tok0, tok1)
    assert result is not None
    # sampled = "10"
    # Case 2: "\n" whitespace -> skipped; "10"+"0"="100"->100, effective=0.9*0.6=0.54
    #         continuation_prob = 0.6
    # Case 1: val=10, stop_prob=(0.6+0.4)-0.6=0.4, effective=0.9*0.4=0.36
    # Non-sampled: "9" standalone, prob=0.1
    expected = (100 * 0.54 + 10 * 0.36 + 9 * 0.1) / (0.54 + 0.36 + 0.1)
    assert abs(result - expected) < 1e-6


def test_two_token_out_of_range_filtered():
    """tok0='50' + tok1='0' would be 500, should be filtered out."""
    tok0 = {"50": math.log(0.9), "30": math.log(0.1)}
    tok1 = {"0": math.log(0.5), "\n": math.log(0.5)}
    result = get_judge_score_two_token(tok0, tok1)
    assert result is not None
    # sampled = "50"
    # Case 2: "\n" whitespace -> skipped; "50"+"0"="500"->500, out of range, skipped
    #         continuation_prob = 0
    # Case 1: val=50, stop_prob=(0.5+0.5)-0=1.0, effective=0.9*1.0=0.9
    # Non-sampled: "30" standalone, prob=0.1
    expected = (50 * 0.9 + 30 * 0.1) / (0.9 + 0.1)
    assert abs(result - expected) < 1e-6


def test_two_token_space_in_tok1_prevents_continuation():
    """tok1=' 5' (with leading space) should not concatenate with tok0 as a number."""
    tok0 = {"9": math.log(0.9), "8": math.log(0.1)}
    tok1 = {" 5": math.log(0.7), "\n": math.log(0.3)}
    result = get_judge_score_two_token(tok0, tok1)
    assert result is not None
    # sampled = "9"
    # Case 2: "\n" whitespace -> skipped; "9"+" 5"="9 5" -> ValueError, skipped
    #         continuation_prob = 0
    # Case 1: val=9, stop_prob=(0.7+0.3)-0=1.0, effective=0.9*1.0=0.9
    # Non-sampled: "8" standalone, prob=0.1
    expected = (9 * 0.9 + 8 * 0.1) / (0.9 + 0.1)
    assert abs(result - expected) < 1e-6


def test_two_token_min_prob_returns_none():
    """Should return None when total probability mass is too low."""
    tok0 = {"hello": math.log(0.9), "world": math.log(0.1)}
    tok1 = {"!": math.log(1.0)}
    result = get_judge_score_two_token(tok0, tok1)
    assert result is None


def test_two_token_non_sampled_out_of_range_filtered():
    """Non-sampled tok0 alternatives outside 0-100 should be filtered."""
    tok0 = {"95": math.log(0.8), "200": math.log(0.1), "-5": math.log(0.1)}
    tok1 = {"\n": math.log(1.0)}
    result = get_judge_score_two_token(tok0, tok1)
    assert result is not None
    # sampled = "95"
    # Case 1: val=95, stop_prob=1.0, effective=0.8*1.0=0.8
    # Non-sampled: "200" out of range, "-5" out of range, both skipped
    assert abs(result - 95.0) < 1e-6


# --- get_judge_score_from_completion tests ---

def test_completion_valid_integer():
    assert get_judge_score_from_completion("95") == 95.0


def test_completion_valid_with_whitespace():
    assert get_judge_score_from_completion("  50  ") == 50.0


def test_completion_boundary_values():
    assert get_judge_score_from_completion("0") == 0.0
    assert get_judge_score_from_completion("100") == 100.0


def test_completion_out_of_range():
    assert get_judge_score_from_completion("101") is None
    assert get_judge_score_from_completion("-1") is None
    assert get_judge_score_from_completion("500") is None


def test_completion_non_numeric():
    assert get_judge_score_from_completion("hello") is None
    assert get_judge_score_from_completion("") is None


def test_completion_float_string():
    assert get_judge_score_from_completion("95.5") == 95.5
