#!/usr/bin/env python3
"""
Comprehensive Financial Sentiment Analysis Tool
HS25 Big Data Assignment - Group Work Part 2

Core module for analyzing sentiment in SEC financial filings using ProsusAI FinBERT.
Provides text chunking, sentiment classification, aggregation, and reporting capabilities.
"""

import json
import re
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration constants
DEFAULT_MODEL_NAME = "ProsusAI/finbert"
DEFAULT_MAX_CHARS = 800
DEFAULT_MAX_SEQUENCE_LENGTH = 512
MIN_SENTENCE_WORDS = 3
MIN_SECTION_LENGTH = 100
SENTIMENT_THRESHOLD = 0.05
HIGH_CONFIDENCE_THRESHOLD = 0.6
LOW_CONFIDENCE_THRESHOLD = 0.3
POSITIVE_RATIO_THRESHOLD = 0.7
NEGATIVE_RATIO_THRESHOLD = 0.6
MIXED_SIGNAL_THRESHOLD = 0.3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinancialSentimentAnalyzer:
    """
    Financial sentiment analyzer using ProsusAI FinBERT model.
    
    FinBERT is a pre-trained NLP model specifically designed for financial text
    sentiment analysis. It provides three-class classification: positive, negative, neutral.
    """
    
    def __init__(self, model_preference: str = "auto", max_length: int = DEFAULT_MAX_SEQUENCE_LENGTH):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_preference: Backend preference ('auto', 'transformers', 'vader')
            max_length: Maximum sequence length for tokenization (default: 512)
        """
        self.model_preference = model_preference
        self.max_length = max_length
        self.use_transformers = False
        self.classifier_pipeline = None
        self.tokenizer = None
        self.model = None
        self.vader_analyzer = None
        
        self._setup_models()
        logger.info(f"Analyzer initialized with backend: {'FinBERT (Transformers)' if self.use_transformers else 'VADER'}")
    
    def _setup_models(self) -> None:
        """Initialize ML models in order of preference."""
        if self.model_preference in ["transformers", "auto"]:
            try:
                from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
                import torch
                
                logger.info(f"Loading FinBERT model: {DEFAULT_MODEL_NAME}")
                
                # Determine device
                device = 0 if torch.cuda.is_available() else -1
                device_name = "GPU" if device == 0 else "CPU"
                logger.info(f"Using device: {device_name}")
                
                # Load tokenizer and model separately for better control
                self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
                self.model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_MODEL_NAME)
                
                # Create pipeline with proper configuration
                self.classifier_pipeline = pipeline(
                    "text-classification",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=device,
                    return_all_scores=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                
                self.use_transformers = True
                logger.info("FinBERT model loaded successfully")
                return
                
            except ImportError as e:
                logger.warning(f"Transformers library not available: {e}. Falling back to VADER.")
            except Exception as e:
                logger.warning(f"Failed to load FinBERT model: {e}. Falling back to VADER.")
        
        if self.model_preference in ["vader", "auto"]:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self.vader_analyzer = SentimentIntensityAnalyzer()
                logger.info("VADER sentiment analyzer loaded successfully")
                return
            except ImportError as e:
                logger.error(f"VADER library not available: {e}")
            except Exception as e:
                logger.error(f"Failed to load VADER: {e}")
        
        raise RuntimeError(
            "No sentiment analysis backend available. "
            "Install transformers library: pip install transformers torch"
        )
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using pattern matching.
        
        Args:
            text: Input text to split
            
        Returns:
            List of sentences
        """
        if not text.strip():
            return []
        
        # Split on sentence-ending punctuation followed by whitespace and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Filter out very short fragments (likely noise)
        sentences = [s for s in sentences if len(s.split()) >= MIN_SENTENCE_WORDS]
        
        return sentences
    
    def chunk_sentences(self, sentences: List[str], max_chars: int = DEFAULT_MAX_CHARS) -> List[str]:
        """
        Group sentences into coherent chunks without exceeding character limit.
        
        Args:
            sentences: List of sentences to chunk
            max_chars: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Handle oversized sentences
            if sentence_length > max_chars:
                # Save existing chunk if any
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split long sentence into word-based sub-chunks
                words = sentence.split()
                sub_chunk = []
                sub_length = 0
                
                for word in words:
                    word_length = len(word) + 1  # Account for space
                    if sub_length + word_length > max_chars:
                        if sub_chunk:
                            chunks.append(" ".join(sub_chunk))
                        sub_chunk = [word]
                        sub_length = word_length
                    else:
                        sub_chunk.append(word)
                        sub_length += word_length
                
                if sub_chunk:
                    chunks.append(" ".join(sub_chunk))
                continue
            
            # Check if adding sentence would exceed limit
            if current_length + sentence_length + 1 > max_chars and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + 1
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        logger.debug(f"Created {len(chunks)} chunks from {len(sentences)} sentences")
        return chunks
    
    def _normalize_finbert_label(self, label: str) -> str:
        """
        Normalize FinBERT label to standard format.
        
        Args:
            label: Raw label from FinBERT
            
        Returns:
            Normalized label (POSITIVE, NEGATIVE, or NEUTRAL)
        """
        label_upper = label.upper()
        if "POSITIVE" in label_upper or label_upper == "POS":
            return "POSITIVE"
        elif "NEGATIVE" in label_upper or label_upper == "NEG":
            return "NEGATIVE"
        else:
            return "NEUTRAL"
    
    def classify_text(self, text: str) -> Dict[str, Any]:
        """
        Classify text sentiment using available backend.
        
        Args:
            text: Text to classify
            
        Returns:
            Dictionary with label, score, and backend information
        """
        if not text.strip():
            return {"label": "NEUTRAL", "score": 0.0, "backend": "none"}
        
        # Use FinBERT if available
        if self.use_transformers and self.classifier_pipeline:
            try:
                # Truncate text if necessary (handled by tokenizer, but limit input for efficiency)
                text_to_analyze = text[:2000]  # Reasonable limit for processing
                
                # Get predictions from FinBERT
                results = self.classifier_pipeline(text_to_analyze)
                
                # FinBERT returns list of all scores, find the highest
                if isinstance(results, list) and len(results) > 0:
                    if isinstance(results[0], list):
                        # Multiple scores format
                        scores = {item['label']: item['score'] for item in results[0]}
                    else:
                        # Single result format
                        scores = {results[0]['label']: results[0]['score']}
                    
                    # Get the label with highest score
                    best_label = max(scores.items(), key=lambda x: x[1])
                    normalized_label = self._normalize_finbert_label(best_label[0])
                    
                    return {
                        "label": normalized_label,
                        "score": best_label[1],
                        "backend": "finbert",
                        "all_scores": scores
                    }
                else:
                    logger.warning("Unexpected FinBERT output format")
                    
            except Exception as e:
                logger.warning(f"FinBERT analysis failed: {e}. Falling back to VADER.")
        
        # Fallback to VADER
        if self.vader_analyzer:
            vs = self.vader_analyzer.polarity_scores(text)
            comp = vs.get("compound", 0.0)
            
            if comp >= SENTIMENT_THRESHOLD:
                label = "POSITIVE"
            elif comp <= -SENTIMENT_THRESHOLD:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"
            
            return {
                "label": label,
                "score": abs(comp),
                "backend": "vader",
                "vader_scores": vs
            }
        
        return {"label": "ERROR", "score": 0.0, "backend": "none"}
    
    def analyze_text(self, text: str, max_chars: int = DEFAULT_MAX_CHARS) -> List[Tuple[int, str, Dict]]:
        """
        Comprehensive text analysis with chunking and sentiment classification.
        
        Args:
            text: Text to analyze
            max_chars: Maximum characters per chunk
            
        Returns:
            List of tuples (chunk_id, chunk_text, sentiment_result)
        """
        if not text:
            logger.warning("Empty text provided for analysis")
            return []
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        logger.info(f"Analyzing text of length {len(text)} characters")
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
        if not paragraphs:
            paragraphs = [text]
        
        # Process each paragraph
        all_chunks = []
        for paragraph in paragraphs:
            sentences = self.split_into_sentences(paragraph)
            chunks = self.chunk_sentences(sentences, max_chars)
            all_chunks.extend(chunks)
        
        # Classify sentiment for each chunk
        results = []
        for idx, chunk in enumerate(all_chunks, 1):
            try:
                sentiment_result = self.classify_text(chunk)
                results.append((idx, chunk, sentiment_result))
                logger.debug(
                    f"Chunk {idx}: {sentiment_result['label']} "
                    f"(score: {sentiment_result['score']:.3f})"
                )
            except Exception as e:
                logger.error(f"Failed to analyze chunk {idx}: {e}")
                results.append((idx, chunk, {"label": "ERROR", "score": 0.0, "backend": "none"}))
        
        return results
    
    def aggregate_sentiment(self, results: List[Tuple[int, str, Dict]]) -> Dict[str, Any]:
        """
        Aggregate sentiment scores from all chunks to determine overall document sentiment.
        
        Args:
            results: List of chunk analysis results
            
        Returns:
            Dictionary with aggregated sentiment metrics
        """
        if not results:
            return {
                "label": "NEUTRAL",
                "score": 0.0,
                "confidence": 0.0,
                "compound_score": 0.0,
                "chunk_count": 0,
                "sentiment_distribution": {"Positive": 0, "Negative": 0, "Neutral": 0, "ERROR": 0},
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
            }
        
        total_weighted = 0.0
        total_score = 0.0
        count = 0
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0, "ERROR": 0}
        
        for _, _, res in results:
            label = str(res.get("label", "")).upper()
            score = float(res.get("score", 0.0) or 0.0)
            
            # Map labels to sentiment categories
            if label.startswith("POS"):
                sign = 1.0
                sentiment_counts["Positive"] += 1
            elif label.startswith("NEG"):
                sign = -1.0
                sentiment_counts["Negative"] += 1
            elif label == "ERROR":
                sign = 0.0
                sentiment_counts["ERROR"] += 1
            else:
                sign = 0.0
                sentiment_counts["Neutral"] += 1
            
            weight = score
            total_weighted += sign * weight
            total_score += score
            count += 1
        
        if count == 0:
            return {
                "label": "NEUTRAL",
                "score": 0.0,
                "confidence": 0.0,
                "compound_score": 0.0,
                "chunk_count": 0,
                "sentiment_distribution": sentiment_counts,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
            }
        
        doc_compound = total_weighted / count
        overall_score = total_score / count
        
        # Calculate ratios
        positive_ratio = sentiment_counts["Positive"] / count
        negative_ratio = sentiment_counts["Negative"] / count
        neutral_ratio = sentiment_counts["Neutral"] / count
        
        # Determine overall label using both distribution and compound score
        # Priority: Clear majority (>50%) OR strong compound score
        if positive_ratio > 0.5:
            # Majority positive chunks
            doc_label = "Positive"
        elif negative_ratio > 0.5:
            # Majority negative chunks
            doc_label = "Negative"
        elif doc_compound >= SENTIMENT_THRESHOLD:
            # Strong positive compound score
            doc_label = "Positive"
        elif doc_compound <= -SENTIMENT_THRESHOLD:
            # Strong negative compound score
            doc_label = "Negative"
        elif positive_ratio > negative_ratio and positive_ratio > 0.3:
            # More positive than negative, and significant positive presence
            doc_label = "Positive"
        elif negative_ratio > positive_ratio and negative_ratio > 0.3:
            # More negative than positive, and significant negative presence
            doc_label = "Negative"
        else:
            # Default to neutral
            doc_label = "Neutral"
        
        confidence = min(overall_score * 1.5, 1.0)
        
        return {
            "label": doc_label,
            "score": abs(doc_compound),
            "confidence": confidence,
            "compound_score": doc_compound,
            "chunk_count": count,
            "sentiment_distribution": sentiment_counts,
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
        }
    
    def generate_report(
        self,
        results: List[Tuple[int, str, Dict]],
        aggregate: Dict[str, Any],
        section_name: str = "Unknown"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report.
        
        Args:
            results: Chunk-level analysis results
            aggregate: Aggregated sentiment metrics
            section_name: Name of the analyzed section
            
        Returns:
            Dictionary containing the full analysis report
        """
        # Find strongly positive chunks
        positive_chunks = [
            (idx, chunk, res)
            for idx, chunk, res in results
            if res.get("label", "").upper().startswith("POS") and res.get("score", 0) > 0.7
        ]
        
        # Find strongly negative chunks
        negative_chunks = [
            (idx, chunk, res)
            for idx, chunk, res in results
            if res.get("label", "").upper().startswith("NEG") and res.get("score", 0) > 0.7
        ]
        
        # Calculate risk indicators
        risk_indicators = {
            "high_optimism_risk": (
                aggregate["positive_ratio"] > POSITIVE_RATIO_THRESHOLD and
                aggregate["confidence"] > HIGH_CONFIDENCE_THRESHOLD
            ),
            "high_pessimism_risk": aggregate["negative_ratio"] > NEGATIVE_RATIO_THRESHOLD,
            "low_confidence": aggregate["confidence"] < LOW_CONFIDENCE_THRESHOLD,
            "mixed_signals": (
                aggregate["positive_ratio"] > MIXED_SIGNAL_THRESHOLD and
                aggregate["negative_ratio"] > MIXED_SIGNAL_THRESHOLD
            ),
        }
        
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "section_analyzed": section_name,
            "overall_sentiment": aggregate,
            "chunk_count": len(results),
            "risk_indicators": risk_indicators,
            "sample_positive_chunks": positive_chunks[:3],
            "sample_negative_chunks": negative_chunks[:3],
            "backend_used": results[0][2].get("backend", "unknown") if results else "none",
        }
        
        return report


def load_filing_data(file_path: str, sections: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Load and parse SEC filing JSON data.
    
    Args:
        file_path: Path to JSON file
        sections: Optional list of section names to extract
        
    Returns:
        Dictionary mapping section names to their content
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        if sections is None:
            sections_data = {
                k: v for k, v in data.items()
                if isinstance(v, str) and len(v.strip()) > MIN_SECTION_LENGTH
            }
        else:
            sections_data = {}
            for section in sections:
                content = data.get(section, "")
                if content and len(content.strip()) > MIN_SECTION_LENGTH:
                    sections_data[section] = content
                else:
                    logger.warning(
                        f"Section {section} not found or too short in {file_path}"
                    )
        
        logger.info(f"Loaded {len(sections_data)} sections from {file_path}")
        return sections_data
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Failed to load filing data from {file_path}: {e}")
        return {}


def create_visualization(
    results: List[Tuple[int, str, Dict]],
    aggregate: Dict[str, Any],
    output_path: Optional[str] = None
) -> None:
    """
    Create visualization charts for sentiment analysis results.
    
    Args:
        results: Chunk-level analysis results
        aggregate: Aggregated sentiment metrics
        output_path: Optional path to save the visualization
    """
    if not results:
        logger.warning("No results to visualize")
        return
    
    labels = [res.get("label", "Unknown") for _, _, res in results]
    scores = [res.get("score", 0) for _, _, res in results]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Financial Sentiment Analysis Results', fontsize=16, fontweight='bold')
    
    # Pie chart: Sentiment distribution
    sentiment_counts = aggregate.get("sentiment_distribution", {})
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    ax1.pie(
        [sentiment_counts.get(k, 0) for k in ["Positive", "Negative", "Neutral"]],
        labels=["Positive", "Negative", "Neutral"],
        colors=colors,
        autopct='%1.1f%%',
        startangle=90
    )
    ax1.set_title('Sentiment Distribution')
    
    # Histogram: Score distribution (confidence scores)
    if scores:
        # Filter out invalid scores and ensure they're in [0, 1] range
        valid_scores = [s for s in scores if 0 <= s <= 1]
        if valid_scores:
            mean_score = sum(valid_scores) / len(valid_scores)
            ax2.hist(valid_scores, bins=20, alpha=0.7, color='#3498db', edgecolor='black', range=(0, 1))
            ax2.set_xlabel('Confidence Score')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Score Distribution')
            ax2.set_xlim(0, 1)
            ax2.axvline(
                mean_score,
                color='red',
                linestyle='--',
                linewidth=2,
                label=f'Mean: {mean_score:.3f}'
            )
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No valid scores available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Score Distribution')
    else:
        ax2.text(0.5, 0.5, 'No scores available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Score Distribution')
    
    # Scatter plot: Sentiment by chunk
    chunk_ids = [idx for idx, _, _ in results]
    sentiment_numeric = []
    for _, _, res in results:
        label = res.get("label", "Neutral").upper()
        if label.startswith("POS"):
            sentiment_numeric.append(1)
        elif label.startswith("NEG"):
            sentiment_numeric.append(-1)
        else:
            sentiment_numeric.append(0)
    
    ax3.scatter(
        chunk_ids,
        sentiment_numeric,
        c=sentiment_numeric,
        cmap='RdYlGn',
        alpha=0.6,
        s=50
    )
    ax3.set_xlabel('Chunk ID')
    ax3.set_ylabel('Sentiment (-1: Neg, 0: Neu, +1: Pos)')
    ax3.set_title('Sentiment by Text Chunk')
    ax3.grid(True, alpha=0.3)
    
    # Bar chart: Sentiment composition
    risk_data = aggregate.get("sentiment_distribution", {})
    risk_labels = list(risk_data.keys())
    risk_values = list(risk_data.values())
    
    bars = ax4.bar(
        risk_labels,
        risk_values,
        color=['#2ecc71', '#e74c3c', '#95a5a6', '#f39c12']
    )
    ax4.set_title('Sentiment Composition')
    ax4.set_ylabel('Number of Chunks')
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom'
        )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_path}")
    else:
        plt.show()


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="Advanced Financial Sentiment Analysis Tool using ProsusAI FinBERT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis of MD&A section
  python sentiment_analysis.py --input-file data/filing.json --section item_7

  # Analyze multiple sections with visualization
  python sentiment_analysis.py --input-file data/filing.json --sections item_1A item_7 --visualize

  # Custom chunk size
  python sentiment_analysis.py --input-file data/filing.json --section item_7 --max-chars 500
        """,
    )
    
    parser.add_argument(
        '--input-file', '-i',
        required=True,
        help='Path to SEC filing JSON file'
    )
    parser.add_argument(
        '--section', '-s',
        default='item_7',
        help='Specific section to analyze (e.g., item_7, item_1A)'
    )
    parser.add_argument(
        '--sections', '-S',
        nargs='+',
        help='Multiple sections to analyze'
    )
    parser.add_argument(
        '--max-chars', '-c',
        type=int,
        default=DEFAULT_MAX_CHARS,
        help=f'Maximum characters per chunk (default: {DEFAULT_MAX_CHARS})'
    )
    parser.add_argument(
        '--model', '-m',
        default='auto',
        choices=['auto', 'transformers', 'vader'],
        help='Sentiment model preference (default: auto)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Detailed output including all chunks'
    )
    parser.add_argument(
        '--visualize', '-V',
        action='store_true',
        help='Generate visualization charts'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output file for results (JSON format)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input_file).exists():
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Initialize analyzer
    try:
        analyzer = FinancialSentimentAnalyzer(model_preference=args.model)
    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        sys.exit(1)
    
    # Determine sections to analyze
    sections_to_analyze = args.sections if args.sections else [args.section]
    
    # Load filing data
    filing_data = load_filing_data(args.input_file, sections_to_analyze)
    if not filing_data:
        logger.error("No analyzable sections found in the filing")
        sys.exit(1)
    
    # Process each section
    all_results = {}
    for section_name, section_text in filing_data.items():
        logger.info(f"Analyzing section: {section_name}")
        
        chunk_results = analyzer.analyze_text(section_text, max_chars=args.max_chars)
        aggregate_sentiment = analyzer.aggregate_sentiment(chunk_results)
        section_report = analyzer.generate_report(chunk_results, aggregate_sentiment, section_name)
        
        all_results[section_name] = {
            "chunk_results": chunk_results,
            "aggregate": aggregate_sentiment,
            "report": section_report,
        }
        
        # Print summary
        print(f"\n{'=' * 60}")
        print(f"ANALYSIS RESULTS: {section_name.upper()}")
        print(f"{'=' * 60}")
        print(f"Overall Sentiment: {aggregate_sentiment['label']}")
        print(f"Confidence Score: {aggregate_sentiment['confidence']:.3f}")
        print(f"Compound Score: {aggregate_sentiment['compound_score']:.3f}")
        print(f"Chunks Analyzed: {aggregate_sentiment['chunk_count']}")
        print(f"Positive Ratio: {aggregate_sentiment['positive_ratio']:.3f}")
        print(f"Negative Ratio: {aggregate_sentiment['negative_ratio']:.3f}")
        
        # Print risk indicators
        risk_indicators = section_report["risk_indicators"]
        print(f"\nRisk Indicators:")
        for indicator, present in risk_indicators.items():
            status = "⚠️ " if present else "✓ "
            print(f"  {status} {indicator}: {present}")
        
        # Verbose output
        if args.verbose and chunk_results:
            print(f"\nDetailed Chunk Analysis:")
            for idx, chunk, res in chunk_results[:5]:
                print(f"\n--- Chunk {idx} ({res.get('backend', 'unknown')}) ---")
                print(f"Sentiment: {res.get('label')} (score: {res.get('score'):.3f})")
                if len(chunk) > 200:
                    print(f"Text: {chunk[:200]}...")
                else:
                    print(f"Text: {chunk}")
    
    # Generate visualization
    if args.visualize and all_results:
        first_section = list(all_results.keys())[0]
        chunk_results = all_results[first_section]["chunk_results"]
        aggregate = all_results[first_section]["aggregate"]
        
        viz_path = args.output.replace('.json', '.png') if args.output else None
        create_visualization(chunk_results, aggregate, viz_path)
    
    # Save results
    if args.output:
        serializable_results = {}
        for section_name, results in all_results.items():
            serializable_results[section_name] = {
                "aggregate": results["aggregate"],
                "report": results["report"],
                "chunk_count": len(results["chunk_results"]),
            }
        
        with open(args.output, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
