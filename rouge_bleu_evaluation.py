# rouge_bleu_evaluation.py - Content Verification using ROUGE/BLEU metrics

import re
import nltk
import json
import numpy as np
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import traceback


# Download required NLTK data (run once)
# nltk.download('punkt')
# nltk.download('stopwords')

class ContentVerificationSystem:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        self.stemmer = PorterStemmer()

    def preprocess_text(self, text):
        """Preprocess text for evaluation"""
        if not text:
            return []

        # Convert to lowercase and tokenize
        try:
            tokens = word_tokenize(text.lower())
        except:
            # Fallback tokenization if NLTK fails
            tokens = re.findall(r'\b\w+\b', text.lower())

        # Remove stopwords and non-alphabetic tokens
        tokens = [token for token in tokens
                  if token.isalpha() and token not in self.stop_words and len(token) > 2]

        # Optional: Apply stemming
        tokens = [self.stemmer.stem(token) for token in tokens]

        return tokens

    def get_ngrams(self, tokens, n):
        """Extract n-grams from token list"""
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    def calculate_rouge_1(self, reference_tokens, candidate_tokens):
        """Calculate ROUGE-1 (unigram) scores"""
        if not reference_tokens or not candidate_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        ref_unigrams = Counter(reference_tokens)
        cand_unigrams = Counter(candidate_tokens)

        # Calculate overlap
        overlap = sum((ref_unigrams & cand_unigrams).values())

        # Precision: overlapping unigrams / candidate unigrams
        precision = overlap / len(candidate_tokens) if candidate_tokens else 0.0

        # Recall: overlapping unigrams / reference unigrams
        recall = overlap / len(reference_tokens) if reference_tokens else 0.0

        # F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }

    def calculate_rouge_2(self, reference_tokens, candidate_tokens):
        """Calculate ROUGE-2 (bigram) scores"""
        if not reference_tokens or not candidate_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        ref_bigrams = Counter(self.get_ngrams(reference_tokens, 2))
        cand_bigrams = Counter(self.get_ngrams(candidate_tokens, 2))

        if not ref_bigrams or not cand_bigrams:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        # Calculate overlap
        overlap = sum((ref_bigrams & cand_bigrams).values())

        # Precision and recall
        precision = overlap / sum(cand_bigrams.values()) if sum(cand_bigrams.values()) > 0 else 0.0
        recall = overlap / sum(ref_bigrams.values()) if sum(ref_bigrams.values()) > 0 else 0.0

        # F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }

    def calculate_rouge_l(self, reference_tokens, candidate_tokens):
        """Calculate ROUGE-L (Longest Common Subsequence) scores"""
        if not reference_tokens or not candidate_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        # Dynamic programming to find LCS length
        def lcs_length(seq1, seq2):
            m, n = len(seq1), len(seq2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if seq1[i - 1] == seq2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

            return dp[m][n]

        lcs_len = lcs_length(reference_tokens, candidate_tokens)

        # Precision and recall
        precision = lcs_len / len(candidate_tokens) if candidate_tokens else 0.0
        recall = lcs_len / len(reference_tokens) if reference_tokens else 0.0

        # F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }

    def calculate_bleu_score(self, reference_tokens, candidate_tokens, max_n=4):
        """Calculate BLEU score with multiple n-gram precision"""
        if not reference_tokens or not candidate_tokens:
            return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0, "bleu_avg": 0.0}

        # Calculate n-gram precisions
        precisions = []

        for n in range(1, min(max_n + 1, len(candidate_tokens) + 1)):
            ref_ngrams = Counter(self.get_ngrams(reference_tokens, n))
            cand_ngrams = Counter(self.get_ngrams(candidate_tokens, n))

            if not cand_ngrams:
                precisions.append(0.0)
                continue

            # Calculate clipped precision
            overlap = sum((ref_ngrams & cand_ngrams).values())
            precision = overlap / sum(cand_ngrams.values())
            precisions.append(precision)

        # Pad with zeros if we don't have enough n-grams
        while len(precisions) < 4:
            precisions.append(0.0)

        # Calculate brevity penalty
        ref_len = len(reference_tokens)
        cand_len = len(candidate_tokens)

        if cand_len == 0:
            bp = 0
        elif cand_len > ref_len:
            bp = 1
        else:
            bp = np.exp(1 - ref_len / cand_len)

        # Calculate BLEU scores (geometric mean of precisions * brevity penalty)
        bleu_scores = {}
        for i in range(4):
            if i < len(precisions) and precisions[i] > 0:
                bleu_scores[f"bleu_{i + 1}"] = round(bp * precisions[i], 4)
            else:
                bleu_scores[f"bleu_{i + 1}"] = 0.0

        # Average BLEU (geometric mean of all non-zero precisions)
        non_zero_precisions = [p for p in precisions if p > 0]
        if non_zero_precisions:
            geo_mean = np.exp(np.mean(np.log(non_zero_precisions)))
            bleu_scores["bleu_avg"] = round(bp * geo_mean, 4)
        else:
            bleu_scores["bleu_avg"] = 0.0

        return bleu_scores

    def evaluate_summary_content(self, transcript, summary):
        """Evaluate summary content against transcript using ROUGE/BLEU"""
        print(f"Evaluating summary against transcript...")

        # Preprocess texts
        ref_tokens = self.preprocess_text(transcript)
        cand_tokens = self.preprocess_text(summary)

        print(f"Reference tokens: {len(ref_tokens)}, Candidate tokens: {len(cand_tokens)}")

        # Calculate all metrics
        rouge_1 = self.calculate_rouge_1(ref_tokens, cand_tokens)
        rouge_2 = self.calculate_rouge_2(ref_tokens, cand_tokens)
        rouge_l = self.calculate_rouge_l(ref_tokens, cand_tokens)
        bleu_scores = self.calculate_bleu_score(ref_tokens, cand_tokens)

        return {
            "summary_evaluation": {
                "rouge_1": rouge_1,
                "rouge_2": rouge_2,
                "rouge_l": rouge_l,
                "bleu": bleu_scores,
                "content_verification": self._interpret_scores(rouge_1, rouge_2, bleu_scores)
            }
        }

    def evaluate_topic_summaries(self, transcript, topic_summary):
        """Evaluate each topic summary against transcript"""
        print(f"Evaluating topic summaries against transcript...")

        if not topic_summary or "topics" not in topic_summary:
            return {"error": "Invalid topic summary structure"}

        topics = topic_summary["topics"]
        ref_tokens = self.preprocess_text(transcript)

        topic_evaluations = {}

        for i, topic in enumerate(topics):
            topic_text = topic.get("summary", "")
            key_points = " ".join(topic.get("key_points", []))
            combined_topic_text = f"{topic_text} {key_points}"

            topic_tokens = self.preprocess_text(combined_topic_text)

            print(f"Evaluating topic {i + 1}: {len(topic_tokens)} tokens")

            # Calculate metrics for this topic
            rouge_1 = self.calculate_rouge_1(ref_tokens, topic_tokens)
            rouge_2 = self.calculate_rouge_2(ref_tokens, topic_tokens)
            rouge_l = self.calculate_rouge_l(ref_tokens, topic_tokens)
            bleu_scores = self.calculate_bleu_score(ref_tokens, topic_tokens)

            topic_evaluations[f"topic_{i + 1}"] = {
                "heading": topic.get("heading", f"Topic {i + 1}"),
                "rouge_1": rouge_1,
                "rouge_2": rouge_2,
                "rouge_l": rouge_l,
                "bleu": bleu_scores,
                "content_verification": self._interpret_scores(rouge_1, rouge_2, bleu_scores)
            }

        # Calculate average scores across all topics
        avg_scores = self._calculate_average_topic_scores(topic_evaluations)

        return {
            "topic_evaluations": topic_evaluations,
            "average_topic_scores": avg_scores
        }

    def _calculate_average_topic_scores(self, topic_evaluations):
        """Calculate average scores across all topics"""
        if not topic_evaluations:
            return {}

        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []
        bleu_scores = []

        for topic_data in topic_evaluations.values():
            rouge_1_scores.append(topic_data["rouge_1"]["f1"])
            rouge_2_scores.append(topic_data["rouge_2"]["f1"])
            rouge_l_scores.append(topic_data["rouge_l"]["f1"])
            bleu_scores.append(topic_data["bleu"]["bleu_avg"])

        return {
            "avg_rouge_1_f1": round(np.mean(rouge_1_scores), 4),
            "avg_rouge_2_f1": round(np.mean(rouge_2_scores), 4),
            "avg_rouge_l_f1": round(np.mean(rouge_l_scores), 4),
            "avg_bleu": round(np.mean(bleu_scores), 4)
        }

    def _interpret_scores(self, rouge_1, rouge_2, bleu_scores):
        """Interpret scores to provide content verification status"""
        rouge_1_f1 = rouge_1["f1"]
        rouge_2_f1 = rouge_2["f1"]
        bleu_avg = bleu_scores["bleu_avg"]

        # Define thresholds
        high_threshold = 0.25
        medium_threshold = 0.15

        medium_threshold_r1 = 0.15  # Lowered from 0.15
        medium_threshold_r2 = 0.08

        # Determine verification status
        if rouge_1_f1 >= high_threshold and rouge_2_f1 >= high_threshold:
            status = "VERIFIED"
            confidence = "High"
        elif rouge_1_f1 >= medium_threshold or rouge_2_f1 >= medium_threshold:
            status = "PARTIALLY_VERIFIED"
            confidence = "Medium"
        else:
            status = "UNVERIFIED"
            confidence = "Low"

        return {
            "status": status,
            "confidence": confidence,
            "explanation": self._get_explanation(status, rouge_1_f1, rouge_2_f1, bleu_avg)
        }

    def _get_explanation(self, status, rouge_1, rouge_2, bleu):
        """Provide explanation for verification status"""
        if status == "VERIFIED":
            return f"Content strongly supported by transcript (ROUGE-1: {rouge_1:.3f}, ROUGE-2: {rouge_2:.3f})"
        elif status == "PARTIALLY_VERIFIED":
            return f"Content partially supported by transcript (ROUGE-1: {rouge_1:.3f}, ROUGE-2: {rouge_2:.3f})"
        else:
            return f"Content poorly supported by transcript - potential hallucination (ROUGE-1: {rouge_1:.3f}, ROUGE-2: {rouge_2:.3f})"

    def comprehensive_content_verification(self, transcript, summary, topic_summary):
        """Perform comprehensive content verification"""
        print("Starting comprehensive content verification...")

        try:
            # Evaluate summary
            summary_eval = self.evaluate_summary_content(transcript, summary)

            # Evaluate topics
            topic_eval = self.evaluate_topic_summaries(transcript, topic_summary)

            # Combine results
            verification_report = {
                "timestamp": "generated_timestamp",
                "summary_verification": summary_eval,
                "topic_verification": topic_eval,
                "overall_assessment": self._generate_overall_assessment(summary_eval, topic_eval)
            }

            print("Content verification completed successfully")
            return verification_report

        except Exception as e:
            print(f"Error in content verification: {traceback.format_exc()}")
            return {
                "error": f"Verification failed: {str(e)}",
                "traceback": traceback.format_exc()
            }

    def _generate_overall_assessment(self, summary_eval, topic_eval):
        """Generate overall assessment of content quality"""
        # Extract key scores
        summary_rouge1 = summary_eval.get("summary_evaluation", {}).get("rouge_1", {}).get("f1", 0)
        summary_rouge2 = summary_eval.get("summary_evaluation", {}).get("rouge_2", {}).get("f1", 0)

        topic_avg_rouge1 = topic_eval.get("average_topic_scores", {}).get("avg_rouge_1_f1", 0)
        topic_avg_rouge2 = topic_eval.get("average_topic_scores", {}).get("avg_rouge_2_f1", 0)

        # Calculate overall scores
        overall_rouge1 = (summary_rouge1 + topic_avg_rouge1) / 2
        overall_rouge2 = (summary_rouge2 + topic_avg_rouge2) / 2

        # Determine overall status
        if overall_rouge1 >= 0.25 and overall_rouge2 >= 0.15:
            overall_status = "HIGH_QUALITY"
            recommendation = "Content is well-supported by transcript"
        elif overall_rouge1 >= 0.15 or overall_rouge2 >= 0.1:
            overall_status = "MODERATE_QUALITY"
            recommendation = "Content is partially supported - review for accuracy"
        else:
            overall_status = "LOW_QUALITY"
            recommendation = "Content may contain hallucinations - manual review required"

        return {
            "overall_rouge_1": round(overall_rouge1, 4),
            "overall_rouge_2": round(overall_rouge2, 4),
            "quality_status": overall_status,
            "recommendation": recommendation,
            "component_scores": {
                "summary_rouge_1": summary_rouge1,
                "summary_rouge_2": summary_rouge2,
                "topics_rouge_1": topic_avg_rouge1,
                "topics_rouge_2": topic_avg_rouge2
            }
        }


# Main function for integration
def verify_content_accuracy(transcript, summary, topic_summary):
    """
    Main function to verify content accuracy using ROUGE/BLEU metrics
    """
    verifier = ContentVerificationSystem()
    return verifier.comprehensive_content_verification(transcript, summary, topic_summary)


# Example usage and testing function
def test_verification_system():
    """Test the verification system with sample data"""

    # Sample data
    sample_transcript = """
    Machine learning is a subset of artificial intelligence that enables computers to learn 
    without being explicitly programmed. It involves algorithms that can identify patterns 
    in data and make predictions. There are three main types of machine learning: supervised 
    learning, unsupervised learning, and reinforcement learning. Supervised learning uses 
    labeled data to train models. Unsupervised learning finds patterns in unlabeled data. 
    Reinforcement learning learns through trial and error with rewards and penalties.
    """

    sample_summary = """
    Machine learning is a branch of AI that allows computers to learn automatically. 
    It uses algorithms to find patterns and make predictions from data. The three main 
    types are supervised, unsupervised, and reinforcement learning.
    """

    sample_topics = {
        "title": "Machine Learning Overview",
        "topics": [
            {
                "heading": "What is Machine Learning",
                "summary": "Machine learning enables computers to learn without explicit programming using pattern recognition algorithms.",
                "key_points": ["Subset of AI", "Pattern recognition", "Automatic learning"]
            },
            {
                "heading": "Types of Machine Learning",
                "summary": "There are three main categories: supervised learning with labeled data, unsupervised learning with unlabeled data, and reinforcement learning through trial and error.",
                "key_points": ["Supervised learning", "Unsupervised learning", "Reinforcement learning"]
            }
        ]
    }

    # Run verification
    result = verify_content_accuracy(sample_transcript, sample_summary, sample_topics)

    print("Verification Results:")
    print(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    test_verification_system()
