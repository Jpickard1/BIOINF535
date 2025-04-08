# Enhanced Paper Retrieval Pipeline for RNAseq/scRNAseq Analysis

This enhanced pipeline retrieves relevant scientific papers based on RNAseq/scRNAseq differential expression (DE) results and gene set enrichment analysis (GSEA) results. It generates optimized search keywords from your biological data and uses the Semantic Scholar API to gather papers that are relevant to your biological system, condition, cell types, and affected genes/gene sets.

## Key Features

- Parses differential expression (DE) results and gene set enrichment analysis (GSEA) results
- Automatically extracts significant genes and pathways
- Generates optimized search keywords based on biological context
- Retrieves relevant papers using the Semantic Scholar API
- **University of Michigan institutional access** for paywalled articles
- Handles API rate limits and provides fallback mechanisms
- Saves paper metadata and generates comprehensive reports
- Downloads available PDFs for retrieved papers
- Supports various input file formats (CSV, TSV, Excel)
- Cell type-specific paper retrieval

## Installation

1. Clone this repository or download the files to your local machine
2. Install the required dependencies:

```bash
pip install requests pandas numpy tqdm semanticscholar
```

## Quick Start

Run the enhanced pipeline with the following command:

```bash
python run_enhanced_pipeline.py --system "skin" --condition "keloid" --species "human" --de-file "path/to/de_results.csv" --gsea-file "path/to/gsea_results.csv"
```

## University of Michigan Institutional Access

To access paywalled articles using University of Michigan credentials:

1. Create a file named `credentials.json` with your UMich credentials:
```json
{
  "username": "your_umich_username",
  "password": "your_umich_password",
  "proxy_url": "https://proxy.lib.umich.edu/login"
}
```

2. Run the pipeline with institutional access enabled:
```bash
python run_enhanced_pipeline.py --system "skin" --condition "keloid" --species "human" --de-file "data.csv" --download-pdfs --use-institutional-access
```

## Documentation

For detailed instructions, please refer to:

- [USER_GUIDE.md](USER_GUIDE.md) - Comprehensive user guide with detailed instructions
- [README.md](README.md) - Basic overview and usage information

## Components

The pipeline consists of the following main components:

1. `semantic_scholar_api_enhanced.py` - Interface for interacting with the Semantic Scholar API
2. `institutional_access.py` - Module for accessing paywalled articles using UMich credentials
3. `keyword_generator.py` - System for generating search keywords from DE and GSEA results
4. `paper_retrieval_pipeline_enhanced.py` - Main pipeline that integrates all components
5. `run_enhanced_pipeline.py` - Command-line interface for running the enhanced pipeline

## Example Commands

### Basic example with DE and GSEA results:

```bash
python run_enhanced_pipeline.py --system "skin" --condition "keloid" --species "human" --de-file "sample_data/sample_de_results.csv" --gsea-file "sample_data/sample_gsea_results.csv"
```

### Example with cell types specified directly:

```bash
python run_enhanced_pipeline.py --system "liver" --condition "fibrosis" --species "mouse" --de-file "data/liver_de_results.csv" --cell-types "hepatocyte" "stellate cell" "Kupffer cell"
```

### Example with PDF downloading and institutional access:

```bash
python run_enhanced_pipeline.py --system "brain" --condition "alzheimer" --species "human" --gsea-file "data/alzheimer_gsea.csv" --download-pdfs --max-pdfs 50 --use-institutional-access
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This pipeline was developed as part of the BIOINF 535 project
- Uses the Semantic Scholar API for paper retrieval
- Includes University of Michigan institutional access support
