#!/usr/bin/env python3
"""
Batch Processing for Multiple Financial Filings

Automates the analysis of multiple SEC financial filings stored as JSON files,
extracting sentiment data from specified sections using ProsusAI FinBERT.
Processes each filing, aggregates sentiment scores, and outputs results to CSV.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sentiment_analysis import FinancialSentimentAnalyzer, load_filing_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_batch(
    input_dir: str,
    output_file: str,
    sections: Optional[List[str]] = None,
    model_preference: str = "auto"
) -> None:
    """
    Process multiple SEC filings in batch mode.
    
    Args:
        input_dir: Directory containing JSON files
        output_file: Output CSV file path
        sections: Optional list of sections to analyze
        model_preference: Model backend preference ('auto', 'transformers', 'vader')
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return
    
    if not input_path.is_dir():
        logger.error(f"Input path is not a directory: {input_dir}")
        return
    
    json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        logger.error(f"No JSON files found in {input_dir}")
        return
    
    logger.info(f"Found {len(json_files)} files to process")
    
    # Initialize analyzer once for all files
    try:
        analyzer = FinancialSentimentAnalyzer(model_preference=model_preference)
    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        return
    
    results = []
    processed_count = 0
    error_count = 0
    
    for file_path in json_files:
        try:
            logger.info(f"Processing {file_path.name} ({processed_count + 1}/{len(json_files)})")
            
            # Extract company identifier from filename
            company_id = file_path.stem.split('_')[0]
            
            # Load filing data
            filing_data = load_filing_data(str(file_path), sections)
            
            if not filing_data:
                logger.warning(f"No analyzable sections found in {file_path.name}")
                continue
            
            # Process each section
            for section_name, section_text in filing_data.items():
                try:
                    chunk_results = analyzer.analyze_text(section_text)
                    aggregate = analyzer.aggregate_sentiment(chunk_results)
                    
                    results.append({
                        'company_id': company_id,
                        'file_name': file_path.name,
                        'section': section_name,
                        'overall_sentiment': aggregate['label'],
                        'sentiment_score': aggregate['score'],
                        'confidence': aggregate['confidence'],
                        'compound_score': aggregate['compound_score'],
                        'chunk_count': aggregate['chunk_count'],
                        'positive_ratio': aggregate['positive_ratio'],
                        'negative_ratio': aggregate['negative_ratio'],
                        'backend_used': (
                            chunk_results[0][2].get('backend', 'unknown')
                            if chunk_results else 'none'
                        )
                    })
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(
                        f"Error processing section {section_name} in {file_path.name}: {e}"
                    )
                    error_count += 1
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing file {file_path.name}: {e}")
            error_count += 1
            continue
    
    # Save results
    if results:
        try:
            df = pd.DataFrame(results)
            
            # Ensure output directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(output_file, index=False)
            logger.info(f"Batch processing complete. Results saved to {output_file}")
            
            # Generate visualizations
            _generate_batch_visualizations(df, output_path.parent)
            
            # Print summary
            _print_batch_summary(df, len(json_files), processed_count, error_count)
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    else:
        logger.error("No results generated")


def _generate_batch_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate visualization charts for batch processing results.
    
    Args:
        df: DataFrame with batch results
        output_dir: Directory to save visualizations
    """
    try:
        sns.set_style("whitegrid")
        
        # Sentiment distribution
        plt.figure(figsize=(10, 6))
        sentiment_counts = df['overall_sentiment'].value_counts()
        sns.barplot(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            palette="viridis"
        )
        plt.title('Sentiment Distribution Across All Filings')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.tight_layout()
        
        dist_path = output_dir / 'sentiment_distribution.png'
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved sentiment distribution chart to {dist_path}")
        
        # Average sentiment by company
        if len(df['company_id'].unique()) > 1:
            plt.figure(figsize=(12, 6))
            avg_scores = df.groupby('company_id')['sentiment_score'].mean().reset_index()
            sns.barplot(
                x='company_id',
                y='sentiment_score',
                data=avg_scores,
                palette="coolwarm"
            )
            plt.title('Average Sentiment Score by Company')
            plt.xlabel('Company ID')
            plt.ylabel('Average Sentiment Score')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            company_path = output_dir / 'average_sentiment_by_company.png'
            plt.savefig(company_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved company sentiment chart to {company_path}")
        
    except Exception as e:
        logger.warning(f"Failed to generate visualizations: {e}")


def _print_batch_summary(
    df: pd.DataFrame,
    total_files: int,
    processed_count: int,
    error_count: int
) -> None:
    """
    Print summary statistics for batch processing.
    
    Args:
        df: DataFrame with results
        total_files: Total number of files processed
        processed_count: Number of successful analyses
        error_count: Number of errors encountered
    """
    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total files found: {total_files}")
    print(f"Successful analyses: {processed_count}")
    print(f"Errors encountered: {error_count}")
    print(f"\nAverage sentiment score: {df['sentiment_score'].mean():.3f}")
    print(f"Average confidence: {df['confidence'].mean():.3f}")
    print(f"\nSentiment distribution:")
    print(df['overall_sentiment'].value_counts())
    print("=" * 60)


def main():
    """Main command-line interface for batch processing."""
    parser = argparse.ArgumentParser(
        description="Batch process multiple SEC filings using ProsusAI FinBERT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files in data directory
  python batch_analysis.py --input-dir data/ --output results.csv

  # Process specific sections
  python batch_analysis.py --input-dir data/ --output results.csv --sections item_7 item_1A
        """
    )
    
    parser.add_argument(
        '--input-dir', '-i',
        required=True,
        help='Input directory containing JSON files'
    )
    parser.add_argument(
        '--output', '-o',
        default='batch_results.csv',
        help='Output CSV file path (default: batch_results.csv)'
    )
    parser.add_argument(
        '--sections', '-s',
        nargs='+',
        default=['item_7', 'item_1A'],
        help='Sections to analyze (default: item_7 item_1A)'
    )
    parser.add_argument(
        '--model', '-m',
        default='auto',
        choices=['auto', 'transformers', 'vader'],
        help='Sentiment model preference (default: auto)'
    )
    
    args = parser.parse_args()
    
    process_batch(
        args.input_dir,
        args.output,
        args.sections,
        args.model
    )


if __name__ == '__main__':
    main()
