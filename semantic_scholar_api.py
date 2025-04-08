#!/usr/bin/env python3
"""
Enhanced Semantic Scholar API Interface for Paper Retrieval Pipeline

This module provides improved functions to interact with the Semantic Scholar API
for retrieving research papers based on keywords related to biological systems,
conditions, cell types, and affected genes/gene sets.
"""

import os
import time
import json
import requests
import pandas as pd
from tqdm import tqdm
from semanticscholar import SemanticScholar

class SemanticScholarInterface:
    """Interface for interacting with the Semantic Scholar API."""
    
    def __init__(self, rate_limit_per_minute=30, output_dir=None, api_key=None):
        """
        Initialize the Semantic Scholar API interface.
        
        Args:
            rate_limit_per_minute (int): Maximum number of requests per minute
            output_dir (str): Directory to save retrieved papers
            api_key (str): Optional API key for Semantic Scholar
        """
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
        
        os.makedirs(self.paper_info_dir, exist_ok=True)
        os.makedirs(self.paper_text_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Initialize paper tracking
        self.retrieved_papers = set()
        self.paper_metadata_file = os.path.join(self.metadata_dir, 'paper_metadata.csv')
        self._load_existing_metadata()
    
    def _load_existing_metadata(self):
        """Load existing paper metadata if available."""
        if os.path.exists(self.paper_metadata_file):
            try:
                metadata_df = pd.read_csv(self.paper_metadata_file)
                self.retrieved_papers = set(metadata_df['paper_id'].tolist())
                print(f"Loaded {len(self.retrieved_papers)} existing paper IDs from metadata.")
            except Exception as e:
                print(f"Error loading existing metadata: {e}")
                self.retrieved_papers = set()
        else:
            self.retrieved_papers = set()
    
    def _respect_rate_limit(self):
        """Ensure API rate limits are respected."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.request_interval:
            sleep_time = self.request_interval - elapsed
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def search_papers(self, query, limit=100, fields=None):
        """
        Search for papers using the Semantic Scholar API.
        
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
                        print(f"Error during search with library: {e}")
                        # Fall back to direct API call
                        break
        except Exception as e:
            print(f"Error with semanticscholar library: {e}")
        
        # If we didn't get any results, try direct API call
        if not papers:
            print(f"Falling back to direct API call for query: {query}")
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
                            print(f"API error: {response.status_code} - {response.text}")
                            time.sleep(5)  # Wait before retrying
                            break
            except Exception as e:
                print(f"Error during direct API call: {e}")
        
        # If still no results, try a simplified query
        if not papers and ' ' in query:
            simplified_query = ' '.join(query.split()[:3])  # Use only first 3 terms
            print(f"Trying simplified query: {simplified_query}")
            
            try:
                with tqdm(total=limit, desc=f"Simplified Search: {simplified_query}") as pbar:
                    self._respect_rate_limit()
                    results = self.sch.search_paper(simplified_query, limit=page_size, fields=fields)
                    
                    if results and len(results) > 0:
                        papers.extend(results)
                        pbar.update(len(results))
            except Exception as e:
                print(f"Error with simplified query: {e}")
        
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
            print(f"Error retrieving paper details for {paper_id}: {e}")
            
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
                    print(f"API error: {response.status_code} - {response.text}")
            except Exception as e2:
                print(f"Error during direct API call for paper details: {e2}")
            
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
            print(f"Error saving paper info for {paper_id}: {e}")
            return False
    
    def _update_metadata(self, paper_data, query=None):
        """Update the metadata CSV with new paper information."""
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
        
        # Create new row
        new_row = {
            'paper_id': paper_id,
            'title': title,
            'authors': author_str,
            'year': year,
            'journal': journal,
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
            print(f"\nProcessing query: {query}")
            papers = self.search_papers(query, limit=limit_per_query, fields=fields)
            
            query_papers = 0
            for paper in papers:
                if self.save_paper_info(paper, query):
                    query_papers += 1
            
            results_summary['queries'][query] = len(papers)
            results_summary['total_papers'] += len(papers)
            
            print(f"Retrieved {len(papers)} papers for query: {query}")
        
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
    
    def download_paper_pdfs(self, max_papers=None):
        """
        Download available PDFs for retrieved papers.
        
        Args:
            max_papers (int): Maximum number of PDFs to download
            
        Returns:
            dict: Summary of download results
        """
        if not os.path.exists(self.paper_metadata_file):
            return {"error": "No papers have been retrieved yet"}
        
        pdf_dir = os.path.join(self.output_dir, 'pdfs')
        os.makedirs(pdf_dir, exist_ok=True)
        
        try:
            df = pd.read_csv(self.paper_metadata_file)
            papers_with_pdf = df[df['has_pdf'] == 1]
            
            if max_papers:
                papers_with_pdf = papers_with_pdf.head(max_papers)
            
            download_results = {
                "total_attempted": len(papers_with_pdf),
                "successful": 0,
                "failed": 0,
                "papers": {}
            }
            
            for idx, paper in tqdm(papers_with_pdf.iterrows(), total=len(papers_with_pdf), desc="Downloading PDFs"):
                paper_id = paper['paper_id']
                
                # Get paper details to get PDF URL
                paper_file = os.path.join(self.paper_info_dir, f"{paper_id}.json")
                
                if os.path.exists(paper_file):
                    with open(paper_file, 'r') as f:
                        paper_data = json.load(f)
                    
                    pdf_url = paper_data.get('openAccessPdf', {}).get('url')
                    
                    if pdf_url:
                        pdf_filename = f"{paper_id}.pdf"
                        pdf_path = os.path.join(pdf_dir, pdf_filename)
                        
                        try:
                            self._respect_rate_limit()
                            response = requests.get(pdf_url, stream=True)
                            
                            if response.status_code == 200:
                                with open(pdf_path, 'wb') as f:
                                    for chunk in response.iter_content(chunk_size=8192):
                                        f.write(chunk)
                                
                                download_results["successful"] += 1
                                download_results["papers"][paper_id] = {
                                    "status": "success",
                                    "path": pdf_path
                                }
                            else:
                                download_results["failed"] += 1
                                download_results["papers"][paper_id] = {
                                    "status": "failed",
                                    "error": f"HTTP error: {response.status_code}"
                                }
                        except Exception as e:
                            download_results["failed"] += 1
                            download_results["papers"][paper_id] = {
                                "status": "failed",
                                "error": str(e)
                            }
                    else:
                        download_results["failed"] += 1
                        download_results["papers"][paper_id] = {
                            "status": "failed",
                            "error": "No PDF URL available"
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
