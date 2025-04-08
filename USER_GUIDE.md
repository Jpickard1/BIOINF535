# Enhanced Paper Retrieval Pipeline User Guide

## Overview

The Enhanced Paper Retrieval Pipeline is a comprehensive tool designed to gather relevant scientific papers based on RNAseq/scRNAseq differential expression (DE) results and gene set enrichment analysis (GSEA) results. This pipeline automatically generates optimized search keywords from your biological data and uses the Semantic Scholar API to retrieve papers relevant to your specific biological system, condition, cell types, and affected genes/gene sets.

A key feature of this enhanced version is the ability to access paywalled articles using University of Michigan credentials, significantly expanding the range of accessible scientific literature.

## Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Input File Formats](#input-file-formats)
4. [University of Michigan Institutional Access](#university-of-michigan-institutional-access)
5. [Command-Line Arguments](#command-line-arguments)
6. [Output Files and Directories](#output-files-and-directories)
7. [Using Results with LLM/RAG Pipeline](#using-results-with-llmrag-pipeline)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Usage](#advanced-usage)

## Installation

### Prerequisites

- Python 3.6 or higher
- pip (Python package installer)

### Step 1: Clone or download the repository

```bash
git clone <repository-url>
cd paper_retrieval_pipeline
```

### Step 2: Install required dependencies

```bash
pip install requests pandas numpy tqdm semanticscholar
```

## Basic Usage

The pipeline can be run with the following basic command:

```bash
python run_enhanced_pipeline.py --system "skin" --condition "keloid" --species "human" --de-file "path/to/de_results.csv" --gsea-file "path/to/gsea_results.csv"
```

This command will:
1. Parse the DE and GSEA results
2. Generate optimized search keywords
3. Retrieve relevant papers using the Semantic Scholar API
4. Save paper metadata and generate a comprehensive report

## Input File Formats

### Differential Expression (DE) Results

The DE results file should contain the following columns (column names are flexible and will be auto-detected):

- Gene identifier (e.g., gene_symbol, gene_id)
- Log2 fold change (e.g., log2fc, log2_fold_change)
- P-value or adjusted p-value (e.g., p_value, padj, fdr)

Example:
```
gene_symbol,log2fc,p_value,adj_p_val
COL1A1,2.5,0.0001,0.001
COL3A1,2.3,0.0002,0.002
ACTA2,1.8,0.0005,0.004
...
```

### Gene Set Enrichment Analysis (GSEA) Results

The GSEA results file should contain the following columns (column names are flexible and will be auto-detected):

- Pathway/term identifier or name (e.g., pathway, term_name)
- P-value or adjusted p-value (e.g., p_value, padj, fdr)
- Genes in pathway (optional, e.g., genes, leading_edge)

Example:
```
pathway,p_value,adj_p_val,genes
KEGG_ECM_RECEPTOR_INTERACTION,0.0001,0.001,"COL1A1,COL3A1,FN1,ITGB1,ITGA5,THBS1"
GO_COLLAGEN_FIBRIL_ORGANIZATION,0.0002,0.002,"COL1A1,COL3A1,SPARC,TGFB1"
...
```

### Cell Types File

The cell types file can be a simple text file with one cell type per line, or a CSV/TSV file with a column containing cell types.

Example:
```
fibroblast
keratinocyte
endothelial cell
melanocyte
immune cell
```

## University of Michigan Institutional Access

A key feature of this enhanced pipeline is the ability to access paywalled articles using University of Michigan credentials.

### Setting Up Credentials

1. Create a file named `credentials.json` in the pipeline directory with the following structure:

```json
{
  "username": "your_umich_username",
  "password": "your_umich_password",
  "proxy_url": "https://proxy.lib.umich.edu/login"
}
```

2. Replace `your_umich_username` and `your_umich_password` with your actual University of Michigan credentials.

3. For security, this file should be excluded from version control (added to .gitignore).

### Using Institutional Access

To use institutional access when running the pipeline, add the `--use-institutional-access` flag:

```bash
python run_enhanced_pipeline.py --system "skin" --condition "keloid" --species "human" --de-file "data.csv" --download-pdfs --use-institutional-access
```

This will:
1. Authenticate with the University of Michigan proxy server
2. Use your credentials to access paywalled articles
3. Download PDFs that would otherwise be inaccessible

### How It Works

The institutional access module:
1. Handles the multi-step authentication process with the University of Michigan weblogin system
2. Maintains an authenticated session for downloading papers
3. Converts regular publisher URLs to proxied URLs for institutional access
4. Automatically navigates through intermediate pages to find and download PDFs

## Command-Line Arguments

### Required Arguments

- `--system`: Biological system (e.g., skin, liver)
- `--condition`: Condition of interest (e.g., keloid, fibrosis)
- `--species`: Species (e.g., human, mouse)

### Optional File Inputs

- `--de-file`: Path to differential expression results file
- `--gsea-file`: Path to gene set enrichment analysis results file
- `--cell-types-file`: Path to file with cell type information
- `--cell-types`: Direct list of cell types (space-separated)

### Pipeline Parameters

- `--output-dir`: Output directory (default: 'paper_retrieval_results')
- `--papers-per-query`: Maximum papers per query (default: 100)
- `--max-total-papers`: Maximum total papers (default: 5000)
- `--file-format`: Input file format ('csv', 'tsv', 'excel') (default: 'csv')

### API and Access Parameters

- `--api-key`: Semantic Scholar API key (optional but recommended)
- `--credentials-file`: Path to institutional credentials JSON file (default: 'credentials.json')

### PDF Download Options

- `--download-pdfs`: Download available PDFs for retrieved papers
- `--max-pdfs`: Maximum number of PDFs to download (default: 100)
- `--use-institutional-access`: Use University of Michigan credentials for paywalled articles

### Logging Options

- `--verbose`: Enable verbose logging

## Output Files and Directories

The pipeline creates the following directory structure:

```
paper_retrieval_results/
├── keywords/
│   └── [species]_[system]_[condition]_keywords.json
│   └── [species]_[system]_[condition]_queries.txt
├── papers/
│   ├── paper_info/
│   │   └── [paper_id].json
│   ├── paper_text/
│   ├── metadata/
│   │   └── paper_metadata.csv
│   │   └── retrieval_summary.json
│   │   └── pdf_download_results.json
│   └── pdfs/
│       └── [paper_id].pdf
├── logs/
│   └── pipeline_log_[timestamp].txt
├── [species]_[system]_[condition]_summary.json
├── [species]_[system]_[condition]_statistics.json
└── retrieval_report_[date].md
```

### Key Output Files

- `paper_metadata.csv`: CSV file containing metadata for all retrieved papers
- `retrieval_summary.json`: Summary of the retrieval process
- `retrieval_report_[date].md`: Human-readable report of the pipeline results
- `[paper_id].json`: JSON files containing detailed information for each paper
- `[paper_id].pdf`: Downloaded PDF files (if available and requested)

## Using Results with LLM/RAG Pipeline

The retrieved papers can be used as input for an LLM with RAG capabilities to generate comprehensive reports. The papers are saved in a structured format that can be easily processed by other pipelines.

To use the results with your LLM/RAG pipeline:

1. Point your RAG pipeline to the `paper_retrieval_results/papers/` directory
2. Use the `paper_metadata.csv` file to filter or prioritize papers
3. Process the paper information from the JSON files in `paper_info/`
4. If PDFs were downloaded, they can be found in the `pdfs/` directory

## Troubleshooting

### No Papers Retrieved

If the pipeline doesn't retrieve any papers, try the following:

1. Simplify your search terms (use fewer keywords)
2. Check if the Semantic Scholar API is working properly
3. Try using an API key to increase rate limits
4. Verify that your DE/GSEA results contain significant genes/pathways

### API Rate Limiting

The Semantic Scholar API has rate limits. If you encounter rate limiting issues:

1. Obtain an API key from Semantic Scholar
2. Use the `--api-key` parameter when running the pipeline
3. Reduce the number of queries or papers per query

### Institutional Access Issues

If you encounter issues with institutional access:

1. Verify your University of Michigan credentials in the credentials.json file
2. Check that your account has proper access permissions
3. Ensure you're using the correct proxy URL
4. Try running with the `--verbose` flag for detailed logging
5. Check the logs for specific authentication errors

## Advanced Usage

### Cell Type-Specific Analysis

For cell type-specific analysis, you can:

1. Provide a file with cell types using `--cell-types-file`
2. Directly specify cell types using `--cell-types`

Example:
```bash
python run_enhanced_pipeline.py --system "liver" --condition "fibrosis" --species "mouse" --de-file "data/liver_de_results.csv" --cell-types "hepatocyte" "stellate cell" "Kupffer cell"
```

### Combining Multiple Data Sources

You can provide both DE and GSEA results to get more comprehensive paper retrieval:

```bash
python run_enhanced_pipeline.py --system "skin" --condition "keloid" --species "human" --de-file "data/de_results.csv" --gsea-file "data/gsea_results.csv" --cell-types-file "data/cell_types.txt"
```

### Downloading PDFs with Institutional Access

To download both open access and paywalled PDFs:

```bash
python run_enhanced_pipeline.py --system "brain" --condition "alzheimer" --species "human" --gsea-file "data/alzheimer_gsea.csv" --download-pdfs --max-pdfs 200 --use-institutional-access
```

### Using a Semantic Scholar API Key

For higher rate limits and more reliable access:

```bash
python run_enhanced_pipeline.py --system "skin" --condition "keloid" --species "human" --de-file "data/de_results.csv" --api-key "YOUR_API_KEY" --download-pdfs
```
