#!/usr/bin/env python3
"""
Enhanced Semantic Scholar API Interface for Paper Retrieval Pipeline

This module provides improved functions to interact with the Semantic Scholar API
for retrieving research papers based on keywords related to biological systems,
conditions, cell types, and affected genes/gene sets. It includes institutional
access support for downloading paywalled articles.
"""

import os
import time
import json
import logging
import requests
import pandas as pd
from tqdm import tqdm
from semanticscholar import SemanticScholar

# Import institutional access module
try:
    from institutional_access import InstitutionalAccess
except ImportError:
    InstitutionalAccess = None

class SemanticScholarInterface:
    """
    Interface for interacting with the Semantic Scholar API.
    
    This class provides methods to search for papers, retrieve paper details,
    save paper information, and download paper PDFs. It includes support for
    institutional access to paywalled articles.
    """
    
    def __init__(self, rate_limit_per_minute=30, output_dir=None, api_key=None, 
                 credentials_file=None, log_level=logging.INFO):
        """
        Initialize the Semantic Scholar API interface.
        
        Args:
            rate_limit_per_minute (int): Maximum number of requests per minute
            output_dir (str): Directory to save retrieved papers
            api_key (str): Optional API key for Semantic Scholar
            credentials_file (str): Path to institutional credentials JSON file
            log_level (int): Logging level
        """
        # Set up logging
        self.logger = self._setup_logger(log_level)
        
        # Initialize Semantic Scholar client
        self.sch = SemanticScholar(api_key=api_key)
        self.rate_limit = rate_limit_per_minute
        self.request_interval = 60.0 / rate_limit_per_minute
        self.last_request_time = 0
        self.api_key = api_key
        
        # Set up output directory
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'retrieved_papers')
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories for different types of data
        self.paper_info_dir = os.path.join(self.output_dir, 'paper_info')
        self.paper_text_dir = os.path.join(self.output_dir, 'paper_text')
        self.metadata_dir = os.path.join(self.output_dir, 'metadata')
        self.pdf_dir = os.path.join(self.output_dir, 'pdfs')
        
        os.makedirs(self.paper_info_dir, exist_ok=True)
        os.makedirs(self.paper_text_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.pdf_dir, exist_ok=True)
        
        # Initialize paper tracking
        self.retrieved_papers = set()
        self.paper_metadata_file = os.path.join(self.metadata_dir, 'paper_metadata.csv')
        self._load_existing_metadata()
        
        # Initialize institutional access if available
        self.inst_access = None
        if InstitutionalAccess is not None:
            try:
                self.inst_access = InstitutionalAccess(credentials_file, log_level)
                self.logger.info("Institutional access module initialized")
            except Exception as e:
                self.logger.error(f"Error initializing institutional access: {e}")
    
    def _setup_logger(self, log_level):
        """
        Set up logger for the Semantic Scholar interface.
        
        Args:
            log_level (int): Logging level
            
        Returns:
            logging.Logger: Configured logger
        """
        logger = logging.getLogger('semantic_scholar_interface')
        logger.setLevel(log_level)
        
        # Create console handler if not already present
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_existing_metadata(self):
        """
        Load existing paper metadata if available.
        
        This method loads previously retrieved paper IDs from the metadata CSV file
        to avoid retrieving the same papers multiple times.
        """
        if os.path.exists(self.paper_metadata_file):
            try:
                metadata_df = pd.read_csv(self.paper_metadata_file)
                self.retrieved_papers = set(metadata_df['paper_id'].tolist())
                self.logger.info(f"Loaded {len(self.retrieved_papers)} existing paper IDs from metadata")
            except Exception as e:
                self.logger.error(f"Error loading existing metadata: {e}")
                self.retrieved_papers = set()
        else:
            self.retrieved_papers = set()
    
    def _respect_rate_limit(self):
        """
        Ensure API rate limits are respected.
        
        This method implements a simple rate limiting mechanism by waiting
        between requests to avoid exceeding the API's rate limits.
        """
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.request_interval:
            sleep_time = self.request_interval - elapsed
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def search_papers(self, query, limit=100, fields=None):
        """
        Search for papers using the Semantic Scholar API.
        
        This method searches for papers matching the given query and returns
        the results. It includes fallback mechanisms for handling API errors.
        
        Args:
            query (str): Search query
            limit (int): Maximum number of papers to retrieve
            fields (list): Fields to include in the response
            
        Returns:
            list: List of paper data dictionaries
        """
        if fields is None:
            fields = [
                'paperId', 'title', 'abstract', 'year', 'authors', 
                'journal', 'venue', 'url', 'citationCount', 'openAccessPdf',
                'fieldsOfStudy', 'tldr'
            ]
        
        # Ensure limit is between 1 and 100
        if limit <= 0:
            limit = 1
        elif limit > 100:
            limit = 100
            
        papers = []
        offset = 0
        page_size = min(100, limit)  # API allows max 100 per request
        
        # First try using the semanticscholar library
        try:
            with tqdm(total=limit, desc=f"Searching: {query}") as pbar:
                while offset < limit:
                    self._respect_rate_limit()
                    
                    try:
                        # Use the semanticscholar library for search
                        results = self.sch.search_paper(query, limit=page_size, offset=offset, fields=fields)
                        
                        if not results or len(results) == 0:
                            break
                        
                        papers.extend(results)
                        pbar.update(len(results))
                        
                        if len(results) < page_size:
                            # No more results available
                            break
                        
                        offset += page_size
                        
                    except Exception as e:
                        self.logger.error(f"Error during search with library: {e}")
                        # Fall back to direct API call
                        break
        except Exception as e:
            self.logger.error(f"Error with semanticscholar library: {e}")
        
        # If we didn't get any results, try direct API call
        if not papers:
            self.logger.info(f"Falling back to direct API call for query: {query}")
            try:
                # Direct API call as fallback
                api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
                
                with tqdm(total=limit, desc=f"API Search: {query}") as pbar:
                    while offset < limit:
                        self._respect_rate_limit()
                        
                        params = {
                            "query": query,
                            "limit": page_size,
                            "offset": offset,
                            "fields": ",".join(fields)
                        }
                        
                        headers = {}
                        if self.api_key:
                            headers["x-api-key"] = self.api_key
                        
                        response = requests.get(api_url, params=params, headers=headers)
                        
                        if response.status_code == 200:
                            data = response.json()
                            results = data.get("data", [])
                            
                            if not results or len(results) == 0:
                                break
                            
                            papers.extend(results)
                            pbar.update(len(results))
                            
                            if len(results) < page_size:
                                # No more results available
                                break
                            
                            offset += page_size
                        else:
                            self.logger.error(f"API error: {response.status_code} - {response.text}")
                            time.sleep(5)  # Wait before retrying
                            break
            except Exception as e:
                self.logger.error(f"Error during direct API call: {e}")
        
        # If still no results, try a simplified query
        if not papers and ' ' in query:
            simplified_query = ' '.join(query.split()[:3])  # Use only first 3 terms
            self.logger.info(f"Trying simplified query: {simplified_query}")
            
            try:
                with tqdm(total=limit, desc=f"Simplified Search: {simplified_query}") as pbar:
                    self._respect_rate_limit()
                    # Ensure limit is valid
                    search_limit = min(100, max(1, limit))
                    results = self.sch.search_paper(simplified_query, limit=search_limit, fields=fields)
                    
                    if results and len(results) > 0:
                        papers.extend(results)
                        pbar.update(len(results))
            except Exception as e:
                self.logger.error(f"Error with simplified query: {e}")
        
        return papers[:limit]
    
    def get_paper_details(self, paper_id, fields=None):
        """
        Get detailed information about a specific paper.
        
        Args:
            paper_id (str): Semantic Scholar Paper ID
            fields (list): Fields to include in the response
            
        Returns:
            dict: Paper details
        """
        if fields is None:
            fields = [
                'paperId', 'title', 'abstract', 'year', 'authors', 
                'journal', 'venue', 'url', 'citationCount', 'openAccessPdf',
                'fieldsOfStudy', 'tldr', 'references', 'citations'
            ]
        
        self._respect_rate_limit()
        
        try:
            paper_details = self.sch.get_paper(paper_id, fields=fields)
            return paper_details
        except Exception as e:
            self.logger.error(f"Error retrieving paper details for {paper_id}: {e}")
            
            # Try direct API call as fallback
            try:
                api_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
                
                params = {
                    "fields": ",".join(fields)
                }
                
                headers = {}
                if self.api_key:
                    headers["x-api-key"] = self.api_key
                
                response = requests.get(api_url, params=params, headers=headers)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    self.logger.error(f"API error: {response.status_code} - {response.text}")
            except Exception as e2:
                self.logger.error(f"Error during direct API call for paper details: {e2}")
            
            return None
    
    def save_paper_info(self, paper_data, query=None):
        """
        Save paper information to disk.
        
        Args:
            paper_data (dict): Paper information
            query (str): Search query that retrieved this paper
            
        Returns:
            bool: Success status
        """
        if not paper_data or 'paperId' not in paper_data:
            return False
        
        paper_id = paper_data['paperId']
        
        # Skip if already retrieved
        if paper_id in self.retrieved_papers:
            return True
        
        # Save paper info as JSON
        paper_file = os.path.join(self.paper_info_dir, f"{paper_id}.json")
        
        try:
            with open(paper_file, 'w') as f:
                json.dump(paper_data, f, indent=2)
            
            # Add to tracking
            self.retrieved_papers.add(paper_id)
            
            # Update metadata
            self._update_metadata(paper_data, query)
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving paper info for {paper_id}: {e}")
            return False
    
    def _update_metadata(self, paper_data, query=None):
        """
        Update the metadata CSV with new paper information.
        
        Args:
            paper_data (dict): Paper information
            query (str): Search query that retrieved this paper
        """
        paper_id = paper_data.get('paperId', '')
        title = paper_data.get('title', '')
        year = paper_data.get('year', '')
        
        # Extract authors (first 3)
        authors = paper_data.get('authors', [])
        author_names = [a.get('name', '') for a in authors[:3]]
        author_str = ', '.join(author_names)
        if len(authors) > 3:
            author_str += ' et al.'
        
        # Extract journal/venue
        journal = paper_data.get('journal', {}).get('name', '')
        if not journal:
            journal = paper_data.get('venue', '')
        
        # Extract URL
        url = paper_data.get('url', '')
        
        # Create new row
        new_row = {
            'paper_id': paper_id,
            'title': title,
            'authors': author_str,
            'year': year,
            'journal': journal,
            'url': url,
            'query': query if query else '',
            'citation_count': paper_data.get('citationCount', 0),
            'has_pdf': 1 if paper_data.get('openAccessPdf', {}).get('url') else 0,
            'retrieval_date': time.strftime('%Y-%m-%d')
        }
        
        # Load existing or create new DataFrame
        if os.path.exists(self.paper_metadata_file):
            try:
                metadata_df = pd.read_csv(self.paper_metadata_file)
                # Check if paper already exists in metadata
                if paper_id not in metadata_df['paper_id'].values:
                    metadata_df = pd.concat([metadata_df, pd.DataFrame([new_row])], ignore_index=True)
            except Exception:
                metadata_df = pd.DataFrame([new_row])
        else:
            metadata_df = pd.DataFrame([new_row])
        
        # Save updated metadata
        metadata_df.to_csv(self.paper_metadata_file, index=False)
    
    def batch_retrieve_papers(self, queries, limit_per_query=100, fields=None):
        """
        Retrieve papers for multiple queries and save results.
        
        Args:
            queries (list): List of search queries
            limit_per_query (int): Maximum papers per query
            fields (list): Fields to include in the response
            
        Returns:
            dict: Summary of retrieved papers
        """
        results_summary = {
            'total_queries': len(queries),
            'total_papers': 0,
            'unique_papers': 0,
            'queries': {}
        }
        
        for query in queries:
            self.logger.info(f"\nProcessing query: {query}")
            papers = self.search_papers(query, limit=limit_per_query, fields=fields)
            
            query_papers = 0
            for paper in papers:
                if self.save_paper_info(paper, query):
                    query_papers += 1
            
            results_summary['queries'][query] = len(papers)
            results_summary['total_papers'] += len(papers)
            
            self.logger.info(f"Retrieved {len(papers)} papers for query: {query}")
        
        results_summary['unique_papers'] = len(self.retrieved_papers)
        
        # Save summary
        summary_file = os.path.join(self.metadata_dir, 'retrieval_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        return results_summary
    
    def get_paper_statistics(self):
        """
        Generate statistics about retrieved papers.
        
        Returns:
            dict: Statistics about the papers
        """
        if not os.path.exists(self.paper_metadata_file):
            return {"error": "No papers have been retrieved yet"}
        
        try:
            df = pd.read_csv(self.paper_metadata_file)
            
            stats = {
                "total_papers": len(df),
                "papers_by_year": df['year'].value_counts().to_dict(),
                "papers_with_pdf": df['has_pdf'].sum(),
                "top_journals": df['journal'].value_counts().head(10).to_dict(),
                "queries_used": df['query'].value_counts().to_dict(),
                "average_citations": df['citation_count'].mean()
            }
            
            return stats
        except Exception as e:
            return {"error": f"Error generating statistics: {e}"}
    
    def download_paper_pdfs(self, max_papers=None, use_institutional_access=False):
        """
        Download available PDFs for retrieved papers.
        
        Args:
            max_papers (int): Maximum number of PDFs to download
            use_institutional_access (bool): Whether to use institutional access for paywalled articles
            
        Returns:
            dict: Summary of download results
        """
        if not os.path.exists(self.paper_metadata_file):
            return {"error": "No papers have been retrieved yet"}
        
        # Check if institutional access is available when requested
        if use_institutional_access and not self.inst_access:
            self.logger.warning("Institutional access requested but not available")
            self.logger.warning("Make sure institutional_access.py is in the same directory")
            self.logger.warning("and credentials.json is properly configured")
            use_institutional_access = False
        
        # Authenticate with institutional access if needed
        if use_institutional_access:
            if not self.inst_access.authenticate():
                self.logger.error("Institutional authentication failed")
                self.logger.error("Check your credentials in credentials.json")
                return {"error": "Institutional authentication failed"}
            else:
                self.logger.info("Institutional authentication successful")
        
        try:
            df = pd.read_csv(self.paper_metadata_file)
            
            # For institutional access, we can try all papers
            # For open access only, filter for papers with PDFs
            if use_institutional_access:
                papers_to_download = df
            else:
                papers_to_download = df[df['has_pdf'] == 1]
            
            if max_papers:
                papers_to_download = papers_to_download.head(max_papers)
            
            download_results = {
                "total_attempted": len(papers_to_download),
                "successful": 0,
                "failed": 0,
                "papers": {},
                "institutional_access_used": use_institutional_access
            }
            
            for idx, paper in tqdm(papers_to_download.iterrows(), total=len(papers_to_download), 
                                  desc="Downloading PDFs"):
                paper_id = paper['paper_id']
                
                # Get paper details to get PDF URL
                paper_file = os.path.join(self.paper_info_dir, f"{paper_id}.json")
                
                if os.path.exists(paper_file):
                    with open(paper_file, 'r') as f:
                        paper_data = json.load(f)
                    
                    pdf_url = paper_data.get('openAccessPdf', {}).get('url')
                    paper_url = paper_data.get('url', '')
                    
                    pdf_filename = f"{paper_id}.pdf"
                    pdf_path = os.path.join(self.pdf_dir, pdf_filename)
                    
                    # Try to download the PDF
                    success = False
                    error_message = ""
                    
                    # First try open access PDF if available
                    if pdf_url:
                        try:
                            self._respect_rate_limit()
                            response = requests.get(pdf_url, stream=True)
                            
                            if response.status_code == 200:
                                with open(pdf_path, 'wb') as f:
                                    for chunk in response.iter_content(chunk_size=8192):
                                        f.write(chunk)
                                
                                success = True
                            else:
                                error_message = f"HTTP error: {response.status_code}"
                        except Exception as e:
                            error_message = str(e)
                    else:
                        error_message = "No open access PDF URL available"
                    
                    # If open access failed and institutional access is enabled, try that
                    if not success and use_institutional_access and paper_url:
                        try:
                            success = self.inst_access.download_paper(paper_url, pdf_path)
                            if not success:
                                error_message = "Institutional access download failed"
                        except Exception as e:
                            error_message = f"Institutional access error: {e}"
                    
                    # Update results
                    if success:
                        download_results["successful"] += 1
                        download_results["papers"][paper_id] = {
                            "status": "success",
                            "path": pdf_path,
                            "institutional_access_used": not bool(pdf_url) and use_institutional_access
                        }
                    else:
                        download_results["failed"] += 1
                        download_results["papers"][paper_id] = {
                            "status": "failed",
                            "error": error_message
                        }
                else:
                    download_results["failed"] += 1
                    download_results["papers"][paper_id] = {
                        "status": "failed",
                        "error": "Paper info file not found"
                    }
            
            # Save download results
            results_file = os.path.join(self.metadata_dir, 'pdf_download_results.json')
            with open(results_file, 'w') as f:
                json.dump(download_results, f, indent=2)
            
            return download_results
            
        except Exception as e:
            return {"error": f"Error downloading PDFs: {e}"}

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the interface
    ss_interface = SemanticScholarInterface(output_dir="retrieved_papers")
    
    # Example search
    query = "keloid fibroblasts single cell RNA-seq"
    papers = ss_interface.search_papers(query, limit=10)
    
    # Save results
    for paper in papers:
        ss_interface.save_paper_info(paper, query)
    
    # Print statistics
    stats = ss_interface.get_paper_statistics()
    print(json.dumps(stats, indent=2))
    
    # Download PDFs with institutional access
    download_results = ss_interface.download_paper_pdfs(max_papers=5, use_institutional_access=True)
    print(f"Downloaded {download_results.get('successful', 0)} PDFs successfully")
