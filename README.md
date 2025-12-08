# Financial Sentiment Analysis Tool

**HS25 Big Data Assignment - Part II: Analyzing Unstructured Data**

Automated sentiment analysis of SEC 10-K filings using FinBERT to detect risks, inconsistencies, and potential misrepresentations in financial narratives.

## 🎯 Assignment Deliverables

**Deliverable 1 - Prototype**
- Interactive Jupyter notebook: `financial_sentiment_analysis.ipynb`
- Fully executable with working demo and visualizations
- Demonstrates single-filing analysis and batch processing

**Deliverable 2 - Presentation**
- Presentation guide: `PRESENTATION_GUIDE.md` (15 min + 5 min Q&A)
- Includes talking points, demo flow, and stakeholder analysis
- Screenshots and backup slides included

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager
- ~2GB disk space (for FinBERT model download on first run)

### Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the interactive notebook
jupyter notebook financial_sentiment_analysis.ipynb
```

**Expected Runtime**: 
- First run: 3-5 minutes (includes model download)
- Subsequent runs: 2-3 minutes

## 📁 Project Structure

```
Sentiment-Analyzer/
├── financial_sentiment_analysis.ipynb   # Main working demo
├── PRESENTATION_GUIDE.md                 # 15-min presentation script
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
└── input/                                # Sample SEC filings (JSON)
    ├── 50471_10K_2021_0001654954-21-010502.json
    └── 8670_10K_2021_0000008670-21-000027.json
```

## 📦 Dependencies

```
torch>=2.0.0                 # PyTorch backend
transformers>=4.30.0         # FinBERT model
vaderSentiment>=3.3.2        # Fallback sentiment analyzer
pandas>=1.5.0                # Data processing
matplotlib>=3.7.0            # Visualization
seaborn>=0.12.0              # Statistical graphics
jupyter>=1.0.0               # Notebook environment
ipython>=8.0.0               # IPython
```

## 🎯 How to Use

### Interactive Demo (Recommended for Assignment)

1. **Start the notebook**:
   ```bash
   jupyter notebook financial_sentiment_analysis.ipynb
   ```

2. **Run cells sequentially**:
   - Cell 1-3: Introduction & setup
   - Cell 4: Load sample SEC filing
   - Cell 5-7: Analyze sentiment per section
   - Cell 8-10: Visualize results
   - Cell 11-14: Risk assessment
   - Cell 15-18: Batch processing

### Command-Line Usage (Optional)

The notebook can be converted to Python scripts if needed, but the interactive notebook format is the primary deliverable.

## 📊 What the Tool Does

### Analysis Workflow
1. **Load** SEC filing JSON (Item 7: MD&A, Item 1A: Risk Factors)
2. **Process** text through FinBERT (finance-specific sentiment model)
3. **Aggregate** sentiment scores per section
4. **Check** cross-section consistency
5. **Flag** risks and inconsistencies
6. **Visualize** results with charts and alerts
3. **Detect** tone inconsistencies between sections
4. **Flag** business risks (over-optimism, distress signals, vague language)
5. **Generate** reports and visualizations

### Key Outputs
- **Sentiment scores** per section (Positive/Negative/Neutral)
- **Consistency checks** (tone alignment across sections)
- **Risk indicators** (high-risk patterns flagged)
- **Batch comparisons** (multiple companies analyzed together)
- **CSV exports** (results for further analysis)

## 🎓 Stakeholder Use Cases

**AUDITORS**: Risk-prioritize filing sections for audit focus
**INVESTORS**: Compare company narratives for consistency and outliers
**REGULATORS**: Monitor 1000+ filings for anomalous disclosure patterns

## 📄 Data Format

Input: JSON files with SEC filing sections
```json
{
  "item_7": "Management Discussion text...",
  "item_1A": "Risk Factors text...",
  ...
}
```

Output: 
- Sentiment scores and confidence levels
- Risk assessment (LOW/MEDIUM/HIGH)
- CSV with comparative metrics

## 🔧 Technical Details

**Architecture**:
- Text chunking (sentences → coherent chunks)
- FinBERT classification (finance-specific sentiment)
- Aggregation (chunk-level → document-level scores)
- Risk detection (flags inconsistencies & patterns)

**Risk Indicators Detected**:
- Over-optimism (positive ratio > 70%)
- Distress signals (negative ratio > 60%)
- Vague language (low confidence < 30%)
- Mixed messaging (conflicting tone within section)

## 📝 Assignment Requirements

✅ **Unstructured Data**: SEC 10-K filings (50-500KB text documents)
✅ **Stakeholders**: Auditors, Investors, Regulators (formalized needs & questions)
✅ **NLP Application**: FinBERT sentiment analysis + risk detection
✅ **Prototype**: Interactive Jupyter notebook (fully executable)
✅ **Deliverables**: 
   - Prototype: `financial_sentiment_analysis.ipynb`
   - Presentation: `PRESENTATION_GUIDE.md` (15 min + 5 min Q&A)
✅ **Code Quality**: Relative paths, reproducible, requires only `pip install -r requirements.txt`
✅ **Live Demo**: Sample data included, auto-discovers files

## 🎓 For Your Presentation

**See `PRESENTATION_GUIDE.md`** for:
- Complete 13-slide structure
- Detailed talking points
- Live demo script
- Backup slides with screenshots

## 📚 Additional Resources

The `Documentations/` folder contains detailed analysis:
- `USE_CASE.md` - Comprehensive use case specification
- `REQUIREMENTS_ANALYSIS.md` - Formal stakeholder requirements
- Other documentation files for reference

---

**HS25 Big Data Assignment - Group Work Part 2**
**Created**: December 2025
