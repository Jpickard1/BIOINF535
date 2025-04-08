#!/usr/bin/env python3
"""
Enhanced Paper Retrieval Pipeline

This script integrates the keyword generator, Semantic Scholar API interface,
and institutional access module to create a complete pipeline for retrieving
relevant papers based on RNAseq/scRNAseq differential expression and gene set
enrichment analysis results.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from tqdm import tqdm

# Import our modules
from keyword_generator import KeywordGenerator
from semantic_scholar_api_enhanced import SemanticScholarInterface

class PaperRetrievalPipeline:
    """
    Pipeline for retrieving relevant papers based on DE and GSEA results.
    
    This class integrates the keyword generator and Semantic Scholar API interface
    to create a complete pipeline for retrieving relevant papers. It includes
    support for institutional access to paywalled articles.
    """
    
    def __init__(self, output_dir=None, log_level=logging.INFO):
        """
        Initialize the paper retrieval pipeline.
        
        Args:
            output_dir (str): Base directory for all pipeline outputs
            log_level (int): Logging level
        """
        # Set up logging
        self.logger = self._setup_logger(log_level)
        
        # Set up base output directory
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'paper_retrieval_results')
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories
        self.keywords_dir = os.path.join(self.output_dir, 'keywords')
        self.papers_dir = os.path.join(self.output_dir, 'papers')
        self.logs_dir = os.path.join(self.output_dir, 'logs')
        
        os.makedirs(self.keywords_dir, exist_ok=True)
        os.makedirs(self.papers_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Initialize components
        self.keyword_generator = KeywordGenerator(output_dir=self.keywords_dir)
        self.semantic_scholar = SemanticScholarInterface(output_dir=self.papers_dir)
        
        # Set up logging
        self.log_file = os.path.join(self.logs_dir, f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        self.log(f"Pipeline initialized with output directory: {self.output_dir}")
    
    def _setup_logger(self, log_level):
        """
        Set up logger for the pipeline.
        
        Args:
            log_level (int): Logging level
            
        Returns:
            logging.Logger: Configured logger
        """
        logger = logging.getLogger('paper_retrieval_pipeline')
        logger.setLevel(log_level)
        
        # Create console handler if not already present
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def log(self, message):
        """
        Log a message to the log file and print to console.
        
        Args:
            message (str): Message to log
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        self.logger.info(message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def run_pipeline(self, biological_system, condition, species, 
                    de_file=None, gsea_file=None, cell_types_file=None,
                    cell_types=None, papers_per_query=100, max_total_papers=5000,
                    file_format='csv', api_key=None, credentials_file=None,
                    download_pdfs=False, max_pdfs=100, use_institutional_access=False):
        """
        Run the complete paper retrieval pipeline.
        
        Args:
            biological_system (str): Biological system (e.g., 'skin', 'liver')
            condition (str): Condition of interest (e.g., 'keloid', 'fibrosis')
            species (str): Species (e.g., 'human', 'mouse')
            de_file (str): Path to differential expression results file
            gsea_file (str): Path to gene set enrichment analysis results file
            cell_types_file (str): Path to file with cell type information
            cell_types (list): Direct list of cell types
            papers_per_query (int): Maximum papers to retrieve per query
            max_total_papers (int): Maximum total papers to retrieve
            file_format (str): Format of input files ('csv', 'tsv', 'excel')
            api_key (str): Semantic Scholar API key
            credentials_file (str): Path to institutional credentials file
            download_pdfs (bool): Whether to download available PDFs
            max_pdfs (int): Maximum number of PDFs to download
            use_institutional_access (bool): Whether to use institutional access
            
        Returns:
            dict: Summary of pipeline results
        """
        self.log(f"Starting pipeline for {species} {biological_system} {condition}")
        
        # Set API key if provided
        if api_key:
            self.semantic_scholar.api_key = api_key
            self.log("Using provided Semantic Scholar API key")
        
        # Step 1: Parse input files
        de_results = None
        gsea_results = None
        
        if de_file:
            self.log(f"Parsing differential expression results from {de_file}")
            de_results = self.keyword_generator.parse_de_results(de_file, format_type=file_format)
            self.log(f"Found {de_results.get('total_significant', 0)} significant genes, selected {de_results.get('selected_genes', 0)}")
        
        if gsea_file:
            self.log(f"Parsing gene set enrichment results from {gsea_file}")
            gsea_results = self.keyword_generator.parse_gsea_results(gsea_file, format_type=file_format)
            self.log(f"Found {gsea_results.get('total_significant', 0)} significant pathways, selected {gsea_results.get('selected_pathways', 0)}")
        
        # Step 2: Extract cell types
        extracted_cell_types = self.keyword_generator.extract_cell_types(cell_types_file, cell_types)
        self.log(f"Using cell types: {', '.join(extracted_cell_types)}")
        
        # Step 3: Generate keywords
        self.log("Generating search keywords")
        keywords = self.keyword_generator.generate_keywords(
            biological_system=biological_system,
            condition=condition,
            species=species,
            de_results=de_results,
            gsea_results=gsea_results,
            cell_types=extracted_cell_types
        )
        
        self.log(f"Generated {len(keywords['combined_queries'])} combined search queries")
        
        # Step 4: Generate additional iterative queries if needed
        all_queries = keywords['combined_queries'].copy()
        
        # Add iterative gene queries if we have many genes
        if de_results and 'significant_genes' in de_results and len(de_results['significant_genes']) > 5:
            self.log("Generating iterative gene queries")
            gene_queries = self.keyword_generator.generate_iterative_gene_queries(
                genes=de_results['significant_genes'],
                biological_system=biological_system,
                condition=condition,
                species=species
            )
            all_queries.extend(gene_queries)
            self.log(f"Added {len(gene_queries)} iterative gene queries")
        
        # Add iterative pathway queries if we have many pathways
        if gsea_results and 'significant_pathways' in gsea_results and len(gsea_results['significant_pathways']) > 3:
            self.log("Generating iterative pathway queries")
            pathway_queries = self.keyword_generator.generate_iterative_pathway_queries(
                pathways=gsea_results['significant_pathways'],
                biological_system=biological_system,
                condition=condition,
                species=species
            )
            all_queries.extend(pathway_queries)
            self.log(f"Added {len(pathway_queries)} iterative pathway queries")
        
        # Step 5: Retrieve papers using Semantic Scholar API
        self.log(f"Starting paper retrieval with {len(all_queries)} queries")
        
        # Calculate papers per query based on max total
        papers_per_query = min(papers_per_query, max_total_papers // len(all_queries) if all_queries else 100)
        self.log(f"Retrieving up to {papers_per_query} papers per query, maximum total: {max_total_papers}")
        
        # Retrieve papers
        retrieval_results = self.semantic_scholar.batch_retrieve_papers(
            queries=all_queries,
            limit_per_query=papers_per_query
        )
        
        # Step 6: Download PDFs if requested
        if download_pdfs:
            self.log(f"Downloading PDFs (max: {max_pdfs}, institutional access: {use_institutional_access})")
            
            # Initialize institutional access if needed
            if use_institutional_access and credentials_file:
                self.log(f"Using institutional access with credentials from {credentials_file}")
            elif use_institutional_access:
                self.log("Using institutional access with default credentials file")
            
            # Download PDFs
            download_results = self.semantic_scholar.download_paper_pdfs(
                max_papers=max_pdfs,
                use_institutional_access=use_institutional_access
            )
            
            self.log(f"Downloaded {download_results.get('successful', 0)} PDFs successfully")
            self.log(f"Failed to download {download_results.get('failed', 0)} PDFs")
            
            # Add download results to retrieval results
            retrieval_results['pdf_download'] = {
                'successful': download_results.get('successful', 0),
                'failed': download_results.get('failed', 0),
                'institutional_access_used': download_results.get('institutional_access_used', False)
            }
        
        # Step 7: Generate summary
        self.log("Paper retrieval completed")
        self.log(f"Retrieved {retrieval_results['total_papers']} total papers, {retrieval_results['unique_papers']} unique papers")
        
        # Generate detailed summary
        summary = {
            'pipeline_info': {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'biological_system': biological_system,
                'condition': condition,
                'species': species,
                'de_file': de_file,
                'gsea_file': gsea_file,
                'cell_types': extracted_cell_types,
                'institutional_access_used': use_institutional_access
            },
            'keyword_stats': {
                'combined_queries': len(keywords['combined_queries']),
                'total_queries': len(all_queries)
            },
            'retrieval_stats': retrieval_results,
            'output_locations': {
                'papers_dir': self.papers_dir,
                'keywords_dir': self.keywords_dir,
                'logs_dir': self.logs_dir,
                'metadata_file': self.semantic_scholar.paper_metadata_file,
                'pdf_dir': self.semantic_scholar.pdf_dir if download_pdfs else None
            }
        }
        
        # Save summary
        summary_file = os.path.join(self.output_dir, f"{species}_{biological_system}_{condition}_summary.json")
        summary_file = summary_file.replace(' ', '_').lower()
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.log(f"Pipeline summary saved to {summary_file}")
        
        # Generate paper statistics
        stats = self.semantic_scholar.get_paper_statistics()
        stats_file = os.path.join(self.output_dir, f"{species}_{biological_system}_{condition}_statistics.json")
        stats_file = stats_file.replace(' ', '_').lower()
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.log(f"Paper statistics saved to {stats_file}")
        
        return summary
    
    def generate_report(self):
        """
        Generate a human-readable report of the pipeline results.
        
        Returns:
            str: Path to the generated report file
        """
        # Check if we have metadata
        if not os.path.exists(self.semantic_scholar.paper_metadata_file):
            self.log("No paper metadata found, cannot generate report")
            return None
        
        try:
            # Load metadata
            metadata_df = pd.read_csv(self.semantic_scholar.paper_metadata_file)
            
            # Create report file
            report_file = os.path.join(self.output_dir, f"retrieval_report_{datetime.now().strftime('%Y%m%d')}.md")
            
            with open(report_file, 'w') as f:
                f.write("# Paper Retrieval Pipeline Report\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Summary statistics
                f.write("## Summary Statistics\n\n")
                f.write(f"- Total papers retrieved: {len(metadata_df)}\n")
                f.write(f"- Publication years: {metadata_df['year'].min()} - {metadata_df['year'].max()}\n")
                f.write(f"- Papers with PDF available: {metadata_df['has_pdf'].sum()}\n")
                f.write(f"- Average citations per paper: {metadata_df['citation_count'].mean():.1f}\n\n")
                
                # Top journals
                f.write("## Top Journals\n\n")
                top_journals = metadata_df['journal'].value_counts().head(10)
                for journal, count in top_journals.items():
                    f.write(f"- {journal}: {count} papers\n")
                f.write("\n")
                
                # Top search queries
                f.write("## Most Productive Search Queries\n\n")
                top_queries = metadata_df['query'].value_counts().head(10)
                for query, count in top_queries.items():
                    f.write(f"- \"{query}\": {count} papers\n")
                f.write("\n")
                
                # Recent highly-cited papers
                f.write("## Recent Highly-Cited Papers\n\n")
                recent_years = metadata_df['year'] >= (metadata_df['year'].max() - 5)
                recent_papers = metadata_df[recent_years].sort_values('citation_count', ascending=False).head(10)
                
                for idx, paper in recent_papers.iterrows():
                    f.write(f"### {paper['title']}\n")
                    f.write(f"- Authors: {paper['authors']}\n")
                    f.write(f"- Year: {paper['year']}\n")
                    f.write(f"- Journal: {paper['journal']}\n")
                    f.write(f"- Citations: {paper['citation_count']}\n")
                    f.write(f"- Paper ID: {paper['paper_id']}\n")
                    if paper['url']:
                        f.write(f"- URL: {paper['url']}\n")
                    f.write("\n")
                
                # PDF download information
                pdf_download_file = os.path.join(self.semantic_scholar.metadata_dir, 'pdf_download_results.json')
                if os.path.exists(pdf_download_file):
                    try:
                        with open(pdf_download_file, 'r') as pdf_file:
                            pdf_results = json.load(pdf_file)
                        
                        f.write("## PDF Download Results\n\n")
                        f.write(f"- Total PDFs attempted: {pdf_results.get('total_attempted', 0)}\n")
                        f.write(f"- Successfully downloaded: {pdf_results.get('successful', 0)}\n")
                        f.write(f"- Failed downloads: {pdf_results.get('failed', 0)}\n")
                        f.write(f"- Institutional access used: {pdf_results.get('institutional_access_used', False)}\n\n")
                    except Exception as e:
                        self.log(f"Error reading PDF download results: {e}")
                
                # Instructions for next steps
                f.write("## Next Steps\n\n")
                f.write("The retrieved papers are saved in the following locations:\n\n")
                f.write(f"- Paper metadata: `{self.semantic_scholar.paper_metadata_file}`\n")
                f.write(f"- Paper information: `{self.semantic_scholar.paper_info_dir}`\n")
                if os.path.exists(self.semantic_scholar.pdf_dir):
                    f.write(f"- Downloaded PDFs: `{self.semantic_scholar.pdf_dir}`\n")
                f.write("\n")
                
                f.write("These files can be used as input for the LLM/RAG pipeline to generate comprehensive reports.\n")
                
                # Institutional access information
                f.write("\n## Institutional Access\n\n")
                f.write("To access paywalled articles using University of Michigan credentials:\n\n")
                f.write("1. Create a `credentials.json` file with your UM credentials\n")
                f.write("2. Run the pipeline with `--use-institutional-access` flag\n")
                f.write("3. Add `--download-pdfs` to download both open access and paywalled articles\n\n")
                f.write("Example credentials.json format:\n")
                f.write("```json\n")
                f.write('{\n  "username": "your_umich_username",\n  "password": "your_umich_password",\n')
                f.write('  "proxy_url": "https://proxy.lib.umich.edu/login"\n}\n')
                f.write("```\n")
            
            self.log(f"Report generated at {report_file}")
            return report_file
            
        except Exception as e:
            self.log(f"Error generating report: {e}")
            return None

# Import pandas here to avoid circular import in the class
import pandas as pd

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
                        help='Use institutional access for paywalled articles')
    
    # Logging options
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging level
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
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
    pipeline = PaperRetrievalPipeline(output_dir=args.output_dir, log_level=log_level)
    
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
    print(f"Results saved to: {args.output_dir}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
