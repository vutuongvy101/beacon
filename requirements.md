# COMP8420 Assignment 3 – Requirements
## Use Case 2: Social Media Intelligent Platform

---

## 1. Project Overview

**Unit:** COMP8420 — 2026 S1  
**Assignment:** Main Project (Problem-based Group Project)  
**Weight:** 40% of Final Grade  
**Use Case:** 2.2 Social Media Intelligent Platform

### Goal

Develop a system that:
- Monitors social media content
- Identifies trends
- Analyzes public sentiments
- Detects emerging topics
- Generates insights for brand monitoring and campaign management

### Suggested Datasets
- [Sentiment140 Twitter Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
- Reddit API (for real-time data)

---

## 2. Functional Requirements

### 2.1 Basic NLP Techniques (minimum 3 required)

| # | Technique | Description |
|---|-----------|-------------|
| B1 | Text Preprocessing Pipeline | Tokenization, normalization, stemming/lemmatization |
| B2 | Word Embedding | Vectorized text representations |
| B3 | Named Entity Recognition (NER) | Entity/location extraction using spaCy, NLTK, or custom model |
| B4 | Rule-based Information Extraction | Hashtag and mention extraction with regex |
| B5 | Sentiment Classification | Sentiment analysis using traditional ML (Naive Bayes, Logistic Regression, BoW/TF-IDF) |
| B6 | Trend Analysis | Identify trending topics over time |
| B7 | Influencer Identification | Surface high-impact accounts/users |
| B8 | Text Clustering | Unsupervised clustering to detect topics (cosine similarity, sentence embeddings) |

### 2.2 Advanced NLP/LLM Techniques (minimum 3 required, aim for 5+)

| # | Technique | Description |
|---|-----------|-------------|
| A1 | LLM Foundation Models | GPT, Gemma, Llama, or similar for generation and analysis |
| A2 | Prompting Engineering | Templates, ReAct, instructions, preference-based prompts |
| A3 | Chain-of-Thought (CoT) Prompting | Multi-step reasoning for insight generation |
| A4 | RAG (Retrieval-Augmented Generation) | Real-time content acquisition, moderation, and trend explanation |
| A5 | Agentic Design | Autonomous agent(s) orchestrating the pipeline |
| A6 | Crisis Detection | Detect and flag negative sentiment spikes or reputational risks |
| A7 | Multi-lingual Analysis | Support content in multiple languages |
| A8 | Branding Strategy Recommendation | LLM-generated recommendations for brand campaigns |
| A9 | Automated Report Generation | Generate structured insight reports automatically |
| A10 | Automated Evaluation (LLM-as-a-Judge) | Self-evaluation of generated outputs |

### 2.3 System Functional Features

- Ingest social media data (batch or real-time)
- Preprocess and clean raw text
- Extract entities, hashtags, and mentions
- Classify sentiment per post/user/topic
- Detect and track trending topics over time
- Identify influential users/accounts
- Generate natural language trend explanations and insights
- Detect potential crises or negative escalation events
- Produce automated reports summarizing findings
- Support multi-lingual content where applicable

---

## 3. Non-Functional Requirements

| Category | Requirement |
|----------|-------------|
| **Usability** | User-friendly interface for input, output, and interaction |
| **Performance** | Efficient processing; demonstrate system throughput |
| **Cohesion** | All basic and advanced techniques smoothly integrated into a single pipeline |
| **Reproducibility** | All results must be fully reproducible from provided code and data |
| **Scalability** | Design should handle varying data volumes |
| **Maintainability** | Structured codebase with README documentation |
| **Academic Integrity** | All code is original; external resources are properly cited |

---

## 4. Data Requirements

- Use **Sentiment140** or **Reddit API** as the primary dataset
- Document the dataset source and collection process
- Submit a data sample (< 5 MB) for verification
- Describe dataset structure, characteristics, and complexities in the report
- Justify dataset selection compared to alternatives

---

## 5. Evaluation Requirements

- Define baseline models for comparison
- Report quantitative evaluation metrics (accuracy, F1, precision, recall, etc.)
- Compare results across techniques and against alternative methods
- Provide qualitative analysis and discussion of limitations
- Include tables and visualizations of experimental results

---

## 6. Deliverables

### 6.1 Source Code (`Codes/`)
- Structured as Jupyter Notebooks (`Notebook1.ipynb`, `Notebook2.ipynb`, …)
- Includes result files and a `README.md` explaining how to run the project
- Hosted in a GitHub repository (private before deadline, public after)

### 6.2 Project Report (`Report/GroupID_Report.pdf`)
- Max 5,000 words, PDF format
- Sections: Introduction, Problem & Task, Roles & Responsibilities, Methodology, Data & Evaluation, Experiment & Analysis, Recommendation & Discussion, Conclusion, References
- Includes system architecture diagrams and workflow descriptions
- Resource links (GitHub, models, video)

### 6.3 Presentation (`Presentation/Presentation.pdf`)
- PowerPoint/PDF submitted to iLearn before **5 June 2026**
- Covers: problem, methodology, NLP techniques, data, results, findings
- 3–5 minutes (overtime incurs mark reduction)
- All group members must present

### 6.4 Video (`Video/`)
- Max 5 minutes demonstrating the full system workflow (input → output)
- Highlights key NLP techniques, insights, and contributions
- All group members contribute to recording
- Provide download link if file size is too large

---

## 7. Grading Breakdown (Total: 40 marks)

### 7.1 Project Implementation (up to 22 marks)

#### Basic Techniques (up to 6 marks)
| Component | Marks | Criteria |
|-----------|-------|----------|
| Justified selection | 1 | ≥3 basic techniques justified (0.5); more techniques with analysis (0.5) |
| Implementation & integration | 3 | Correct implementation (2); successful integration with analysis (1) |
| Evaluation & analysis | 2 | Quantitative results (1); comparison against alternatives with limitations discussion (1) |

#### Advanced Techniques (up to 11 marks)
| Component | Marks | Criteria |
|-----------|-------|----------|
| LLM foundation models | 2 | Redeveloped (2) / fine-tuned (1.5) / pretrained adoption (0.5) |
| Justified selection of LLM techniques | 2 | ≥5 LLM techniques justified (1); more with analysis (1) |
| Implementation & integration | 4 | Effective LLM implementation (3); integration with basic techniques (1) |
| Evaluation & analysis | 3 | Quantitative results (1); limitations & improvement (1); comparison with other models (1) |

#### System Integration & Performance (up to 5 marks)
| Component | Marks | Criteria |
|-----------|-------|----------|
| User interfaces | 1 | User-friendly I/O and interaction |
| Cohesive design | 1 | All techniques effectively integrated |
| System performance | 1 | Efficient and effective operations |
| Operations | 1 | Smooth end-to-end workflow |
| User experience | 1 | Positive UX |

### 7.2 Project Report (up to 11 marks)

| Component | Marks |
|-----------|-------|
| System architecture & workflow | 1 |
| Data selection & preparation | 1 |
| Comparative analysis | 1 |
| Results presentation | 2 |
| Evaluation & analysis | 2 |
| Critical discussion & reflection | 1 |
| Report organization | 1 |
| Presentation quality | 2 |

### 7.3 Presentation (up to 5 marks)

| Component | Marks |
|-----------|-------|
| Content coverage | 2 |
| System demonstration | 1 |
| Delivery | 0.5 |
| Teamwork | 0.5 |
| Individual contributions | 0.5 |
| Q & A | 0.5 |

### 7.4 Video (up to 2 marks)

| Component | Marks |
|-----------|-------|
| Function demonstration | 1 |
| Interpretation / all members contribute | 1 |

---

## 8. Key Deadlines

| Milestone | Date |
|-----------|------|
| Presentation (Week 13 Workshop) | Friday, 5 June 2026 |
| Presentation file submitted to iLearn | Friday, 5 June 2026 |
| Report, Code & Video submission | Friday, 19 June 2026 (Exam Period) |

**Late policy:** 5% penalty per day, zero after 7 days.

---

## 9. Submission Structure

```
GroupID_Assignment3/
├── Report/
│   └── GroupID_Report.pdf
├── Presentation/
│   └── Presentation.pdf
├── Codes/
│   ├── Notebook1.ipynb
│   ├── Notebook2.ipynb
│   ├── README.md
│   └── <result files>
└── Video/
    └── <video files>
```

Submit as a **single ZIP file** to iLearn.

---

## 10. Academic Integrity Notes

- All code must be original work with proper citations for external resources
- Direct use of LLM platforms (ChatGPT, Gemini) for code generation is **prohibited** and will incur heavy penalty
- Set GitHub repo to **private before the submission deadline**, then make it **public immediately after**
- Do not recycle materials from past or concurrent courses
- Plagiarism checks will be performed
