"""
Example usage of the Sentiment Analysis Toolkit for Financial Reports

# Prerequisites:
pip install torch transformers vaderSentiment matplotlib pandas seaborn

# Single file analysis
python sentiment_analysis.py --input-file data/723531_10K_2021_0000723531-21-000035.json --section item_7 --verbose --visualize

# Batch processing
python batch_analysis.py --input-dir data/ --output output/results.json --visualize

# Comparative analysis
python comparative_analysis.py --company-a data/company1.json --company-b data/company2.json

"""