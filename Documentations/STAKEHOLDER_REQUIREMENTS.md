# Stakeholder Requirements Documentation
## Financial Sentiment Analyzer - HS25 Big Data Assignment

---

## 1. Stakeholders

### 1.1 Financial Auditors
**Who they are:**
- Certified Public Accountants (CPAs) working for audit firms
- Internal auditors within companies
- Regulatory auditors from agencies like the SEC

**Analytical/Information Needs:**
- Detect potential misstatements or inconsistencies in financial reporting
- Identify overly optimistic or pessimistic language that may indicate risks
- Flag discrepancies between different sections of the same filing
- Assess the tone and consistency of management's discussion

**Context of Use:**
- Quarterly and annual audit reviews
- Pre-audit risk assessment
- Compliance verification processes
- Fraud detection investigations

---

### 1.2 Investment Analysts
**Who they are:**
- Equity research analysts at investment banks
- Portfolio managers at asset management firms
- Due diligence analysts at private equity firms
- Credit analysts at rating agencies

**Analytical/Information Needs:**
- Identify risk signals and red flags in company disclosures
- Compare sentiment across multiple companies in a sector
- Track sentiment trends over time for portfolio companies
- Assess management's confidence and transparency

**Context of Use:**
- Investment due diligence processes
- Portfolio monitoring and risk assessment
- Sector analysis and benchmarking
- Credit rating decisions

---

### 1.3 Compliance Officers
**Who they are:**
- Regulatory compliance officers at financial institutions
- Risk management professionals
- Legal counsel reviewing disclosures
- Corporate governance specialists

**Analytical/Information Needs:**
- Monitor regulatory compliance in financial disclosures
- Ensure consistency across reporting periods
- Identify potential disclosure violations
- Track changes in risk factor language

**Context of Use:**
- Ongoing compliance monitoring
- Pre-filing review processes
- Regulatory examination preparation
- Corporate governance assessments

---

### 1.4 Financial Researchers and Academics
**Who they are:**
- Academic researchers studying financial reporting
- Policy researchers analyzing disclosure trends
- Data scientists working with financial text data

**Analytical/Information Needs:**
- Large-scale sentiment analysis across multiple filings
- Trend analysis over time periods
- Cross-sectional comparisons
- Pattern detection in financial language

**Context of Use:**
- Academic research projects
- Policy analysis studies
- Machine learning model training
- Text mining research

---

## 2. Triggers: Contexts for Accessing the Corpus

### 2.1 Quarterly Audit Season
**When:** End of each fiscal quarter (Q1, Q2, Q3, Q4)
**Stakeholders:** Financial Auditors
**Trigger:** Need to review 10-K filings for audit clients and assess risk levels
**Frequency:** Quarterly, with peak activity during annual audit season

---

### 2.2 Investment Due Diligence Process
**When:** Before making investment decisions or portfolio changes
**Stakeholders:** Investment Analysts, Portfolio Managers
**Trigger:** Need to assess company risk profile and management transparency
**Frequency:** On-demand, triggered by investment opportunities

---

### 2.3 Regulatory Compliance Review
**When:** Ongoing monitoring and pre-filing review periods
**Stakeholders:** Compliance Officers, Legal Counsel
**Trigger:** Need to ensure filings meet regulatory standards and identify potential issues
**Frequency:** Continuous monitoring, with intensive review before filing deadlines

---

### 2.4 Risk Assessment for Portfolio Companies
**When:** Regular portfolio monitoring cycles (monthly/quarterly)
**Stakeholders:** Investment Analysts, Risk Managers
**Trigger:** Need to identify changes in risk profile or early warning signals
**Frequency:** Monthly or quarterly reviews

---

### 2.5 Academic Research Projects
**When:** During research study periods
**Stakeholders:** Financial Researchers, Academics
**Trigger:** Need to analyze large datasets of financial disclosures for research
**Frequency:** Project-based, can involve hundreds or thousands of filings

---

## 3. Analytical Questions

### 3.1 Sentiment Analysis Questions

**Q1: What is the overall sentiment tone of the Management's Discussion and Analysis (MD&A) section?**
- **Stakeholders:** All stakeholders
- **Purpose:** Understand management's overall outlook and confidence level
- **Expected Output:** Sentiment label (Positive/Negative/Neutral) with confidence score

**Q2: Are there inconsistencies between the Risk Factors section and the MD&A section?**
- **Stakeholders:** Auditors, Compliance Officers
- **Purpose:** Detect potential red flags where management downplays risks in MD&A but lists them in Risk Factors
- **Expected Output:** Comparative sentiment analysis showing discrepancies

**Q3: Which companies show high pessimism indicators in their filings?**
- **Stakeholders:** Investment Analysts, Risk Managers
- **Purpose:** Identify companies with elevated risk levels or distress signals
- **Expected Output:** Ranked list of companies by pessimism score with risk flags

**Q4: What are the sentiment trends across multiple filings for a specific company?**
- **Stakeholders:** Investment Analysts, Auditors
- **Purpose:** Track changes in management tone over time
- **Expected Output:** Time series visualization showing sentiment evolution

**Q5: How does this company's sentiment compare to industry peers?**
- **Stakeholders:** Investment Analysts, Researchers
- **Purpose:** Benchmark company disclosure tone against sector norms
- **Expected Output:** Comparative analysis with industry averages

---

### 3.2 Risk Detection Questions

**Q6: Are there signs of excessive optimism that might indicate overstatement?**
- **Stakeholders:** Auditors, Compliance Officers
- **Purpose:** Flag potential misrepresentation or overconfidence
- **Expected Output:** Risk indicator flagging "High Optimism Risk" with specific examples

**Q7: Does the filing show mixed signals (both high positive and negative sentiment)?**
- **Stakeholders:** All stakeholders
- **Purpose:** Identify inconsistent messaging that may indicate uncertainty or obfuscation
- **Expected Output:** "Mixed Signals" risk indicator with distribution metrics

**Q8: What specific text passages show the strongest negative sentiment?**
- **Stakeholders:** Investment Analysts, Auditors
- **Purpose:** Identify key risk areas mentioned in the filing
- **Expected Output:** Extracted negative sentiment chunks with context

**Q9: Is the language consistent and confident, or does it show low confidence patterns?**
- **Stakeholders:** Auditors, Compliance Officers
- **Purpose:** Detect hedging language or uncertainty that may indicate problems
- **Expected Output:** Confidence score with "Low Confidence" risk indicator if applicable

---

### 3.3 Batch Analysis Questions

**Q10: Which companies in our portfolio have the highest risk scores based on sentiment?**
- **Stakeholders:** Portfolio Managers, Risk Managers
- **Purpose:** Prioritize review efforts and identify portfolio risks
- **Expected Output:** Ranked CSV/Excel file with risk scores for all companies

**Q11: What percentage of filings show negative sentiment trends year-over-year?**
- **Stakeholders:** Researchers, Policy Analysts
- **Purpose:** Understand broader market trends and disclosure patterns
- **Expected Output:** Aggregate statistics and trend visualizations

**Q12: Can we identify patterns in sentiment across different industries?**
- **Stakeholders:** Researchers, Investment Analysts
- **Purpose:** Sector analysis and comparative benchmarking
- **Expected Output:** Industry-level sentiment aggregations and comparisons

---

## 4. Expected Results: Interaction Facility and Output

### 4.1 Interaction Facilities

#### 4.1.1 Command-Line Interface (CLI)
**For:** Technical users, batch processing, automation
**Features:**
- Single file analysis: `python sentiment_analysis.py --input-file data/filing.json --section item_7`
- Batch processing: `python batch_analysis.py --input-dir data/ --output results.csv`
- Customizable parameters (chunk size, model selection, sections)
- Verbose output for detailed inspection

**Use Cases:**
- Automated batch processing of hundreds of filings
- Integration into existing workflows
- Scripted analysis pipelines

---

#### 4.1.2 Jupyter Notebook Interface
**For:** Interactive analysis, exploration, demonstrations
**Features:**
- Step-by-step analysis workflow
- Interactive visualization
- Comparative analysis across sections
- Risk assessment demonstrations
- Educational and presentation purposes

**Use Cases:**
- Ad-hoc analysis of specific filings
- Exploratory data analysis
- Live demonstrations to stakeholders
- Research and development

---

#### 4.1.3 Programmatic API (Python Module)
**For:** Integration into custom applications
**Features:**
- `FinancialSentimentAnalyzer` class for direct use
- `load_filing_data()` function for data loading
- `process_batch()` function for batch operations
- Customizable analysis parameters

**Use Cases:**
- Integration into proprietary systems
- Custom dashboard development
- Automated reporting systems

---

### 4.2 Output Formats

#### 4.2.1 JSON Reports (Single File Analysis)
**Content:**
- Overall sentiment label (Positive/Negative/Neutral)
- Confidence score (0-1)
- Compound sentiment score
- Chunk-level results with individual sentiment labels
- Sentiment distribution (counts and ratios)
- Risk indicators (flags for high optimism, pessimism, mixed signals, low confidence)
- Sample positive and negative text chunks
- Metadata (file name, sections analyzed, model used)

**Use Cases:**
- Detailed analysis reports
- Integration with other systems
- Audit documentation
- Research data storage

**Example Structure:**
```json
{
  "file_name": "837852_10K_2020_0001104659-21-044740.json",
  "section": "item_7",
  "overall_sentiment": "Neutral",
  "confidence": 0.65,
  "compound_score": 0.12,
  "sentiment_distribution": {
    "Positive": 15,
    "Negative": 12,
    "Neutral": 23
  },
  "risk_indicators": ["Mixed Signals"],
  "chunks": [...]
}
```

---

#### 4.2.2 CSV Reports (Batch Processing)
**Content:**
- Company identifier (CIK)
- File name
- Section analyzed
- Overall sentiment label
- Sentiment score
- Confidence level
- Compound score
- Chunk count
- Positive/negative ratios
- Backend used (transformers/vader)

**Use Cases:**
- Portfolio-wide analysis
- Comparative studies
- Excel-based reporting
- Database import
- Trend analysis

**Example Columns:**
```
company_id,file_name,section,overall_sentiment,sentiment_score,confidence,compound_score,chunk_count,positive_ratio,negative_ratio,backend_used
```

---

#### 4.2.3 Visualizations
**Types:**
- Pie charts: Sentiment distribution
- Bar charts: Comparative analysis across sections
- Histograms: Score distributions
- Scatter plots: Confidence vs. sentiment score

**Use Cases:**
- Presentation materials
- Dashboard displays
- Quick visual assessment
- Stakeholder communication

---

#### 4.2.4 Risk Alerts
**Format:** Text-based warnings with specific indicators
**Content:**
- ⚠️ HIGH OPTIMISM - Potential overstatement risk
- ⚠️ HIGH PESSIMISM - Potential distress indicators
- ⚠️ LOW CONFIDENCE - Inconsistent language detected
- ⚠️ MIXED SIGNALS - Conflicting sentiment patterns

**Use Cases:**
- Quick risk assessment
- Prioritization of review efforts
- Alert systems
- Compliance monitoring

---

### 4.3 Expected User Experience

#### 4.3.1 For Auditors
**Workflow:**
1. Load filing(s) for audit client
2. Analyze MD&A and Risk Factors sections
3. Review sentiment scores and risk indicators
4. Examine flagged inconsistencies
5. Export report for audit documentation

**Key Outputs:**
- Risk indicator flags
- Comparative section analysis
- Confidence scores for assessment reliability

---

#### 4.3.2 For Investment Analysts
**Workflow:**
1. Batch process portfolio companies or sector
2. Review ranked risk scores
3. Drill down into high-risk companies
4. Compare against industry benchmarks
5. Export CSV for further analysis

**Key Outputs:**
- Ranked risk scores
- Comparative analysis
- Trend visualizations
- Exportable data for integration

---

#### 4.3.3 For Compliance Officers
**Workflow:**
1. Process company's own filings before submission
2. Review risk indicators and inconsistencies
3. Identify potential compliance issues
4. Generate compliance reports

**Key Outputs:**
- Consistency checks
- Risk flags
- Compliance-ready reports

---

#### 4.3.4 For Researchers
**Workflow:**
1. Batch process large dataset (hundreds/thousands of filings)
2. Export aggregated results
3. Perform statistical analysis
4. Generate visualizations for publications

**Key Outputs:**
- Large-scale CSV exports
- Aggregate statistics
- Trend analysis data
- Research-ready datasets

---

## 5. Use Case Alignment with Assignment Requirements

### 5.1 Text Mining ✅
- **Requirement:** "text mining"
- **Implementation:** Sentiment analysis extracts patterns and insights from unstructured financial text
- **Evidence:** FinBERT model trained specifically for financial text mining

### 5.2 Clustering of Information ✅
- **Requirement:** "clustering of information"
- **Implementation:** Sentiment-based clustering (Positive/Negative/Neutral) and risk-based grouping
- **Evidence:** Risk indicators create clusters (high optimism, high pessimism, mixed signals)

### 5.3 Efficient Processing ✅
- **Requirement:** "efficient processing or filtering of large amount of text"
- **Implementation:** Batch processing, intelligent chunking, dual backend (fast VADER fallback)
- **Evidence:** Processes 191+ filings efficiently via batch_analysis.py

### 5.4 Getting Meaningful Insights ✅
- **Requirement:** "getting meaningful insights"
- **Implementation:** Risk indicators, sentiment trends, comparative analysis
- **Evidence:** Actionable risk flags and confidence scores

### 5.5 RAG with LLMs ⚠️ (Potential Enhancement)
- **Requirement:** "Retrieval Augmented Generation (RAG) with LLMs"
- **Current Status:** Not implemented (sentiment analysis focus)
- **Potential Enhancement:** Could add LangChain + LLM for question-answering over filings
- **Note:** Current implementation focuses on sentiment analysis, which is a valid NLP use case

---

## 6. Summary

This Financial Sentiment Analyzer addresses the analytical needs of multiple stakeholder groups who require insights from large collections of SEC 10-K financial filings. The system provides:

- **Automated sentiment analysis** using state-of-the-art NLP models
- **Risk detection** through intelligent pattern recognition
- **Scalable batch processing** for large datasets
- **Multiple interaction modes** (CLI, notebook, programmatic)
- **Comprehensive outputs** (JSON, CSV, visualizations, alerts)

The solution enables stakeholders to efficiently process, analyze, and gain insights from financial text corpora that would be impractical to review manually, thereby enhancing decision-making, risk assessment, and compliance monitoring capabilities.
