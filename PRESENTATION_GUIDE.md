# Financial Sentiment Analysis Presentation Guide

**HS25 Big Data Assignment - Part II: Analyzing Unstructured Data**  
*Automated sentiment analysis of SEC 10-K filings using FinBERT to detect risks, inconsistencies, and potential misrepresentations in financial narratives.*

## 🎯 Presentation Structure (15 minutes + 5 minutes Q&A)

### 1. Introduction (2 minutes)
- **Problem Statement**: Financial disclosures contain vast unstructured text that stakeholders need to analyze for sentiment consistency and risk signals.
- **Solution Overview**: NLP-powered sentiment analysis of SEC 10-K filings (Item 7: MD&A, Item 1A: Risk Factors) using FinBERT.
- **Demo Preview**: Live notebook execution showing single-filing and batch analysis.

### 2. Stakeholder Analysis (3 minutes)
- **Audit & Assurance**: Need to verify MD&A claims align with Risk Factors during annual audits.
- **Investor Relations**: Monitor tone consistency with financial performance for earnings communications.
- **Risk & Compliance**: Continuous monitoring for distress signals and mixed messaging.

### 3. Technical Implementation (4 minutes)
- **Data Source**: SEC 10-K JSON filings (narrative sections extracted).
- **NLP Pipeline**: 
  - Text chunking (~800 chars)
  - FinBERT sentiment classification (Positive/Negative/Neutral)
  - Section-level aggregation and consistency checks
- **Fallback Strategy**: VADER for CPU-only environments.
- **Execution**: Relative paths, pip installable, runs in ~3-5 minutes.

### 4. Live Demo (4 minutes)
- **Single Filing Analysis**:
  - Load sample 10-K filing
  - Analyze sentiment per section
  - Visualize results with risk flags
- **Batch Processing**:
  - Process multiple filings
  - Generate summary reports
- **Interactive Features**: Adjustable thresholds, section comparisons.

### 5. Results & Impact (2 minutes)
- **Key Outputs**: Risk-ranked sections, sentiment deltas, flagged passages.
- **Business Value**: Early detection of inconsistencies, improved audit efficiency, enhanced investor confidence.

## 📊 Demo Screenshots (Include in Slides)

### Slide 1: Stakeholder Requirements Table
```
| Stakeholder | Trigger | Key Questions | Expected Result |
|-------------|---------|---------------|-----------------|
| Audit & Assurance | Annual audit planning | Are MD&A claims aligned with Risk Factors? | Risk-ranked sections with evidence |
| Investor Relations | Earnings prep | Is tone consistent with financial performance? | Sentiment deltas by section |
| Risk & Compliance | Continuous monitoring | Are there distress signals? | Alerts with severity (High/Med/Low) |
```

### Slide 2: Technical Architecture
- Flowchart: JSON Load → Text Chunking → FinBERT Analysis → Aggregation → Risk Scoring
- Dependencies: transformers, torch, pandas, matplotlib, seaborn

### Slide 3: Sample Analysis Output
- Sentiment scores: MD&A (Positive: 65%, Neutral: 25%, Negative: 10%)
- Risk Factors (Positive: 20%, Neutral: 30%, Negative: 50%)
- Consistency Check: High inconsistency flag due to tone mismatch

### Slide 4: Visualization Examples
- Bar charts showing sentiment distribution
- Heatmap of section-by-section comparisons
- Risk flag dashboard with color-coded alerts

### Slide 5: Batch Processing Results
- Summary table across multiple filings
- Trend analysis showing sentiment evolution
- Export capabilities (CSV for further analysis)

## 💡 Talking Points

### For Lecturers:
- **Scalability**: Handles large text corpora efficiently
- **Accuracy**: FinBERT trained on financial text vs. general sentiment models
- **Practicality**: Executable on standard hardware with CPU fallback
- **Extensibility**: Framework for other NLP tasks (clustering, RAG)

### For Q&A:
- **Model Choice**: FinBERT vs. general BERT - domain-specific training
- **Limitations**: Sentiment ≠ factual accuracy; requires human validation
- **Future Enhancements**: Multi-lingual support, real-time monitoring
- **Ethical Considerations**: Transparency in AI-driven financial analysis

## 🎨 Slide Design Tips
- Use company colors (if applicable) or professional blue/green palette
- Include code snippets in monospace font
- Add transition animations for demo flow
- Backup slides: Detailed code walkthrough, error handling, performance metrics

## 📈 Expected Demo Flow
1. Start notebook kernel
2. Load sample filing (show JSON structure)
3. Run sentiment analysis (show progress bars)
4. Display visualizations (zoom in on key charts)
5. Demonstrate batch processing
6. Show export functionality

*Note: Prepare 2-3 backup filings in input/ directory for extended demos*</content>
<parameter name="filePath">/Users/annageiser/Desktop/Sentiment-Analyzer/PRESENTATION_GUIDE.md