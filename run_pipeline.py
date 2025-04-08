#!/usr/bin/env python3
"""
Paper Retrieval Pipeline - Main Script

This script provides a command-line interface to run the complete paper retrieval
pipeline for gathering relevant papers based on RNAseq/scRNAseq differential
expression and gene set enrichment analysis results.
"""

import os
import sys
import json
import argparse
from datetime import datetime

from paper_retrieval_pipeline import PaperRetrievalPipeline

def main():
    """Main function to run the pipeline from command line."""
    parser = argparse.ArgumentParser(
        description='Paper Retrieval Pipeline for RNAseq/scRNAseq DE and GSEA results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--system', required=True, 
                        help='Biological system (e.g., skin, liver)')
    parser.add_argument('--condition', required=True, 
                        help='Condition of interest (e.g., keloid, fibrosis)')
    parser.add_argument('--species', required=True, 
                        help='Species (e.g., human, mouse)')
    
    # Optional file inputs
    parser.add_argument('--de-file', 
                        help='Path to differential expression results file')
    parser.add_argument('--gsea-file', 
                        help='Path to gene set enrichment analysis results file')
    parser.add_argument('--cell-types-file', 
                        help='Path to file with cell type information')
    
    # Direct cell types input
    parser.add_argument('--cell-types', nargs='+', 
                        help='List of cell types')
    
    # Pipeline parameters
    parser.add_argument('--output-dir', default='paper_retrieval_results', 
                        help='Output directory')
    parser.add_argument('--papers-per-query', type=int, default=100, 
                        help='Maximum papers per query')
    parser.add_argument('--max-total-papers', type=int, default=5000, 
                        help='Maximum total papers')
    parser.add_argument('--file-format', choices=['csv', 'tsv', 'excel'], default='csv', 
                        help='Input file format')
    parser.add_argument('--api-key', 
                        help='Semantic Scholar API key (optional but recommended)')
    parser.add_argument('--download-pdfs', action='store_true', 
                        help='Download available PDFs for retrieved papers')
    parser.add_argument('--max-pdfs', type=int, default=100, 
                        help='Maximum number of PDFs to download')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "="*80)
    print(" "*20 + "PAPER RETRIEVAL PIPELINE FOR RNASEQ/SCRNASEQ ANALYSIS")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Biological System: {args.system}")
    print(f"Condition: {args.condition}")
    print(f"Species: {args.species}")
    print("-"*80 + "\n")
    
    # Initialize and run pipeline
    pipeline = PaperRetrievalPipeline(output_dir=args.output_dir)
    
    # Set API key if provided
    if args.api_key:
        pipeline.semantic_scholar.api_key = args.api_key
    
    # Run the pipeline
    summary = pipeline.run_pipeline(
        biological_system=args.system,
        condition=args.condition,
        species=args.species,
        de_file=args.de_file,
        gsea_file=args.gsea_file,
        cell_types_file=args.cell_types_file,
        cell_types=args.cell_types,
        papers_per_query=args.papers_per_query,
        max_total_papers=args.max_total_papers,
        file_format=args.file_format
    )
    
    # Download PDFs if requested
    if args.download_pdfs:
        print("\nDownloading available PDFs...")
        download_results = pipeline.semantic_scholar.download_paper_pdfs(max_papers=args.max_pdfs)
        print(f"Downloaded {download_results.get('successful', 0)} PDFs successfully")
    
    # Generate report
    report_path = pipeline.generate_report()
    if report_path:
        print(f"\nGenerated report at: {report_path}")
    
    print("\n" + "="*80)
    print(f"Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Retrieved {summary['retrieval_stats']['unique_papers']} unique papers")
    print(f"Results saved to: {args.output_dir}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
