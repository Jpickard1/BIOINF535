#!/usr/bin/env python3
"""
Enhanced Paper Retrieval Pipeline - Main Script

This script provides a user-friendly command-line interface to run the enhanced
paper retrieval pipeline with institutional access support for University of Michigan.
"""

import os
import sys
import argparse
from datetime import datetime

# Import the enhanced pipeline
from paper_retrieval_pipeline_enhanced import PaperRetrievalPipeline

def main():
    """
    Main function to run the enhanced pipeline from command line.
    
    This function parses command-line arguments, initializes the pipeline,
    and runs it with the specified parameters. It includes support for
    institutional access to paywalled articles using University of Michigan
    credentials.
    """
    parser = argparse.ArgumentParser(
        description='Enhanced Paper Retrieval Pipeline for RNAseq/scRNAseq Analysis with University of Michigan Access',
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
    
    # API and access parameters
    parser.add_argument('--api-key', 
                        help='Semantic Scholar API key (optional but recommended)')
    parser.add_argument('--credentials-file', default='credentials.json',
                        help='Path to institutional credentials JSON file')
    
    # PDF download options
    parser.add_argument('--download-pdfs', action='store_true', 
                        help='Download available PDFs for retrieved papers')
    parser.add_argument('--max-pdfs', type=int, default=100, 
                        help='Maximum number of PDFs to download')
    parser.add_argument('--use-institutional-access', action='store_true',
                        help='Use University of Michigan credentials for paywalled articles')
    
    # Logging options
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "="*80)
    print(" "*15 + "ENHANCED PAPER RETRIEVAL PIPELINE WITH UMICH ACCESS")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Biological System: {args.system}")
    print(f"Condition: {args.condition}")
    print(f"Species: {args.species}")
    
    if args.use_institutional_access:
        print("Using University of Michigan institutional access")
        
        # Check if credentials file exists
        if not os.path.exists(args.credentials_file):
            print(f"\nWARNING: Credentials file '{args.credentials_file}' not found.")
            print("A template will be created. Please edit it with your UMich credentials.")
            print("Run the pipeline again after updating the credentials file.")
            
            # Create a template credentials file
            template = {
                "username": "your_umich_username",
                "password": "your_umich_password",
                "proxy_url": "https://proxy.lib.umich.edu/login"
            }
            
            import json
            try:
                with open(args.credentials_file, 'w') as f:
                    json.dump(template, f, indent=2)
                print(f"Created template credentials file at {args.credentials_file}")
            except Exception as e:
                print(f"Error creating template credentials file: {e}")
                return
    
    print("-"*80 + "\n")
    
    # Initialize and run pipeline
    pipeline = PaperRetrievalPipeline(output_dir=args.output_dir)
    
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
        file_format=args.file_format,
        api_key=args.api_key,
        credentials_file=args.credentials_file,
        download_pdfs=args.download_pdfs,
        max_pdfs=args.max_pdfs,
        use_institutional_access=args.use_institutional_access
    )
    
    # Generate report
    report_path = pipeline.generate_report()
    if report_path:
        print(f"\nGenerated report at: {report_path}")
    
    print("\n" + "="*80)
    print(f"Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Retrieved {summary['retrieval_stats']['unique_papers']} unique papers")
    
    if args.download_pdfs:
        pdf_stats = summary['retrieval_stats'].get('pdf_download', {})
        print(f"Downloaded {pdf_stats.get('successful', 0)} PDFs successfully")
        if args.use_institutional_access:
            print("Used University of Michigan institutional access for paywalled articles")
    
    print(f"Results saved to: {args.output_dir}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
