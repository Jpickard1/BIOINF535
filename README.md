# Paper Retrieval Pipeline for RNAseq/scRNAseq Analysis

This pipeline retrieves relevant scientific papers based on RNAseq/scRNAseq differential expression (DE) results and gene set enrichment analysis (GSEA) results. It generates optimized search keywords from your biological data and uses the Semantic Scholar API to gather papers that are relevant to your biological system, condition, cell types, and affected genes/gene sets.

## Features

- Parses differential expression (DE) results and gene set enrichment analysis (GSEA) results
- Automatically extracts significant genes and pathways
- Generates optimized search keywords based on biological context
- Retrieves relevant papers using the Semantic Scholar API
- Handles API rate limits and provides fallback mechanisms
- Saves paper metadata and generates comprehensive reports
- Downloads available PDFs for retrieved papers
- Supports various input file formats (CSV, TSV, Excel)

## Installation

1. Clone this repository or download the files to your local machine
2. Install the required dependencies:

```bash
pip install requests pandas numpy tqdm semanticscholar
```

## Usage

### Basic Usage

Run the pipeline with the following command:

```bash
python run_pipeline.py --system "skin" --condition "keloid" --species "human" --de-file "path/to/de_results.csv" --gsea-file "path/to/gsea_results.csv"
```

### Required Arguments

- `--system`: Biological system (e.g., skin, liver)
- `--condition`: Condition of interest (e.g., keloid, fibrosis)
- `--species`: Species (e.g., human, mouse)

### Optional Arguments

- `--de-file`: Path to differential expression results file
- `--gsea-file`: Path to gene set enrichment analysis results file
- `--cell-types-file`: Path to file with cell type information
- `--cell-types`: Direct list of cell types (space-separated)
- `--output-dir`: Output directory (default: 'paper_retrieval_results')
- `--papers-per-query`: Maximum papers per query (default: 100)
- `--max-total-papers`: Maximum total papers (default: 5000)
- `--file-format`: Input file format ('csv', 'tsv', 'excel') (default: 'csv')
- `--api-key`: Semantic Scholar API key (optional but recommended)
- `--download-pdfs`: Download available PDFs for retrieved papers
- `--max-pdfs`: Maximum number of PDFs to download (default: 100)

### Example Commands

#### Basic example with DE and GSEA results:

```bash
python run_pipeline.py --system "skin" --condition "keloid" --species "human" --de-file "sample_data/sample_de_results.csv" --gsea-file "sample_data/sample_gsea_results.csv"
```

#### Example with cell types specified directly:

```bash
python run_pipeline.py --system "liver" --condition "fibrosis" --species "mouse" --de-file "data/liver_de_results.csv" --cell-types "hepatocyte" "stellate cell" "Kupffer cell"
```

#### Example with PDF downloading:

```bash
python run_pipeline.py --system "brain" --condition "alzheimer" --species "human" --gsea-file "data/alzheimer_gsea.csv" --download-pdfs --max-pdfs 50
```

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

## Output

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

## Using the Results with LLM/RAG Pipeline

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

## Components

The pipeline consists of the following main components:

1. `semantic_scholar_api.py`: Interface for interacting with the Semantic Scholar API
2. `keyword_generator.py`: System for generating search keywords from DE and GSEA results
3. `paper_retrieval_pipeline.py`: Main pipeline that integrates the components
4. `run_pipeline.py`: Command-line interface for running the pipeline

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This pipeline was developed as part of the BIOINF 535 project
- Uses the Semantic Scholar API for paper retrieval
