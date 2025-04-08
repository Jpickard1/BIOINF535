#!/usr/bin/env python3
"""
Test Script for Enhanced Paper Retrieval Pipeline

This script tests the enhanced paper retrieval pipeline with institutional access
to verify that all components are working correctly.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('pipeline_test')

def test_institutional_access(credentials_file=None):
    """
    Test the institutional access module with University of Michigan credentials.
    
    Args:
        credentials_file (str): Path to credentials JSON file
        
    Returns:
        bool: True if test was successful
    """
    logger.info("Testing institutional access module...")
    
    try:
        from institutional_access import InstitutionalAccess
        
        # Initialize institutional access
        inst_access = InstitutionalAccess(credentials_file)
        
        # Test authentication
        logger.info("Testing authentication with University of Michigan...")
        auth_result = inst_access.test_authentication()
        
        if auth_result:
            logger.info("✓ Authentication successful")
            return True
        else:
            logger.error("✗ Authentication failed")
            return False
            
    except ImportError:
        logger.error("✗ Could not import institutional_access module")
        return False
    except Exception as e:
        logger.error(f"✗ Error testing institutional access: {e}")
        return False

def test_semantic_scholar_api():
    """
    Test the Semantic Scholar API interface.
    
    Returns:
        bool: True if test was successful
    """
    logger.info("Testing Semantic Scholar API interface...")
    
    try:
        from semantic_scholar_api_enhanced import SemanticScholarInterface
        
        # Initialize interface
        ss_interface = SemanticScholarInterface(output_dir="test_output")
        
        # Test search
        logger.info("Testing paper search...")
        query = "keloid fibroblasts single cell RNA-seq"
        papers = ss_interface.search_papers(query, limit=3)
        
        if papers and len(papers) > 0:
            logger.info(f"✓ Successfully retrieved {len(papers)} papers")
            
            # Test saving paper info
            logger.info("Testing paper info saving...")
            saved = ss_interface.save_paper_info(papers[0], query)
            
            if saved:
                logger.info("✓ Successfully saved paper info")
                return True
            else:
                logger.error("✗ Failed to save paper info")
                return False
        else:
            logger.error("✗ Failed to retrieve papers")
            return False
            
    except ImportError:
        logger.error("✗ Could not import semantic_scholar_api_enhanced module")
        return False
    except Exception as e:
        logger.error(f"✗ Error testing Semantic Scholar API: {e}")
        return False

def test_keyword_generator():
    """
    Test the keyword generator with sample data.
    
    Returns:
        bool: True if test was successful
    """
    logger.info("Testing keyword generator...")
    
    try:
        from keyword_generator import KeywordGenerator
        
        # Initialize keyword generator
        kg = KeywordGenerator(output_dir="test_output")
        
        # Test with mock data
        de_results = {
            'significant_genes': ['COL1A1', 'COL3A1', 'ACTA2', 'FN1', 'TGFB1'],
            'upregulated_genes': ['COL1A1', 'COL3A1', 'ACTA2'],
            'downregulated_genes': ['KRT1', 'KRT10']
        }
        
        gsea_results = {
            'significant_pathways': ['KEGG_ECM_RECEPTOR_INTERACTION', 'GO_COLLAGEN_FIBRIL_ORGANIZATION', 
                                    'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION']
        }
        
        cell_types = ['fibroblast', 'keratinocyte', 'endothelial cell']
        
        # Generate keywords
        logger.info("Generating keywords...")
        keywords = kg.generate_keywords(
            biological_system='skin',
            condition='keloid',
            species='human',
            de_results=de_results,
            gsea_results=gsea_results,
            cell_types=cell_types
        )
        
        if keywords and 'combined_queries' in keywords and len(keywords['combined_queries']) > 0:
            logger.info(f"✓ Successfully generated {len(keywords['combined_queries'])} queries")
            return True
        else:
            logger.error("✗ Failed to generate keywords")
            return False
            
    except ImportError:
        logger.error("✗ Could not import keyword_generator module")
        return False
    except Exception as e:
        logger.error(f"✗ Error testing keyword generator: {e}")
        return False

def test_pipeline_integration(credentials_file=None):
    """
    Test the complete pipeline integration.
    
    Args:
        credentials_file (str): Path to credentials JSON file
        
    Returns:
        bool: True if test was successful
    """
    logger.info("Testing complete pipeline integration...")
    
    try:
        from paper_retrieval_pipeline_enhanced import PaperRetrievalPipeline
        
        # Create test output directory
        os.makedirs("test_output", exist_ok=True)
        
        # Initialize pipeline
        pipeline = PaperRetrievalPipeline(output_dir="test_output")
        
        # Test with sample data
        logger.info("Running pipeline with sample data...")
        
        # Check if sample data exists
        sample_de_file = "sample_data/sample_de_results.csv"
        sample_gsea_file = "sample_data/sample_gsea_results.csv"
        sample_cell_types_file = "sample_data/sample_cell_types.txt"
        
        if not os.path.exists(sample_de_file) or not os.path.exists(sample_gsea_file):
            logger.error("✗ Sample data files not found")
            return False
        
        # Run pipeline with minimal paper retrieval to speed up test
        summary = pipeline.run_pipeline(
            biological_system="skin",
            condition="keloid",
            species="human",
            de_file=sample_de_file,
            gsea_file=sample_gsea_file,
            cell_types_file=sample_cell_types_file,
            papers_per_query=2,
            max_total_papers=10,
            credentials_file=credentials_file,
            download_pdfs=False,
            use_institutional_access=False  # Set to True to test institutional access
        )
        
        if summary and 'retrieval_stats' in summary:
            logger.info(f"✓ Pipeline integration test successful")
            logger.info(f"  Retrieved {summary['retrieval_stats'].get('unique_papers', 0)} unique papers")
            return True
        else:
            logger.error("✗ Pipeline integration test failed")
            return False
            
    except ImportError as e:
        logger.error(f"✗ Could not import required modules: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Error testing pipeline integration: {e}")
        return False

def main():
    """Main function to run all tests."""
    parser = argparse.ArgumentParser(description='Test Enhanced Paper Retrieval Pipeline')
    parser.add_argument('--credentials-file', default='credentials.json',
                        help='Path to institutional credentials JSON file')
    parser.add_argument('--test-institutional-access', action='store_true',
                        help='Test institutional access with University of Michigan credentials')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(" "*20 + "ENHANCED PIPELINE TEST SUITE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*80 + "\n")
    
    # Create test output directory
    os.makedirs("test_output", exist_ok=True)
    
    # Run tests
    tests = {
        "Semantic Scholar API": test_semantic_scholar_api(),
        "Keyword Generator": test_keyword_generator(),
        "Pipeline Integration": test_pipeline_integration()
    }
    
    # Test institutional access if requested
    if args.test_institutional_access:
        if os.path.exists(args.credentials_file):
            tests["Institutional Access"] = test_institutional_access(args.credentials_file)
        else:
            print(f"\nWARNING: Credentials file '{args.credentials_file}' not found.")
            print("Skipping institutional access test.")
            print("Create a credentials.json file with your UMich credentials to test this feature.")
    
    # Print test summary
    print("\n" + "-"*80)
    print("TEST SUMMARY")
    print("-"*80)
    
    all_passed = True
    for test_name, result in tests.items():
        status = "PASSED" if result else "FAILED"
        status_color = "\033[92m" if result else "\033[91m"  # Green for pass, red for fail
        print(f"{status_color}{status}\033[0m - {test_name}")
        if not result:
            all_passed = False
    
    print("\n" + "="*80)
    overall_status = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
    overall_color = "\033[92m" if all_passed else "\033[91m"
    print(f"{overall_color}{overall_status}\033[0m")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
