#!/usr/bin/env python3
"""
Keyword Generator for Paper Retrieval Pipeline

This module provides functions to generate search keywords from RNAseq/scRNAseq
differential expression (DE) results and gene set enrichment analysis (GSEA) results.
"""

import os
import json
import pandas as pd
import numpy as np
from collections import Counter

class KeywordGenerator:
    """Generate search keywords from DE and GSEA results."""
    
    def __init__(self, output_dir=None):
        """
        Initialize the keyword generator.
        
        Args:
            output_dir (str): Directory to save generated keywords
        """
        # Set up output directory
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'generated_keywords')
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def parse_de_results(self, file_path, format_type='csv', p_value_threshold=0.05, 
                         log2fc_threshold=1.0, max_genes=50):
        """
        Parse differential expression results file.
        
        Args:
            file_path (str): Path to DE results file
            format_type (str): File format ('csv', 'tsv', 'excel')
            p_value_threshold (float): P-value significance threshold
            log2fc_threshold (float): Log2 fold change threshold
            max_genes (int): Maximum number of genes to include
            
        Returns:
            dict: Parsed DE results with significant genes
        """
        try:
            # Load the file based on format
            if format_type.lower() == 'csv':
                df = pd.read_csv(file_path)
            elif format_type.lower() == 'tsv':
                df = pd.read_csv(file_path, sep='\t')
            elif format_type.lower() == 'excel':
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            # Detect column names for p-value and log2FC
            # Common column names for p-value
            p_value_cols = ['p_val', 'p_value', 'pvalue', 'p.value', 'p-value', 'p_val_adj', 
                           'p_adj', 'padj', 'adj.p.val', 'adj_p_val', 'fdr', 'q_value']
            
            # Common column names for log2FC
            log2fc_cols = ['log2fc', 'log2_fc', 'log2foldchange', 'log2_fold_change', 
                          'log2.fold.change', 'l2fc', 'logfc']
            
            # Common column names for gene identifiers
            gene_cols = ['gene', 'gene_id', 'gene_name', 'gene_symbol', 'symbol', 'ensembl_id']
            
            # Find the actual column names in the dataframe
            p_val_col = next((col for col in p_value_cols if col.lower() in [c.lower() for c in df.columns]), None)
            log2fc_col = next((col for col in log2fc_cols if col.lower() in [c.lower() for c in df.columns]), None)
            gene_col = next((col for col in gene_cols if col.lower() in [c.lower() for c in df.columns]), None)
            
            if not all([p_val_col, log2fc_col, gene_col]):
                # If automatic detection fails, try to infer from column data types
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if not p_val_col and len(numeric_cols) >= 2:
                    # P-values are typically between 0 and 1
                    for col in numeric_cols:
                        if df[col].dropna().between(0, 1).all():
                            p_val_col = col
                            break
                
                if not log2fc_col and len(numeric_cols) >= 2:
                    # log2FC typically has both positive and negative values
                    for col in numeric_cols:
                        if col != p_val_col and df[col].dropna().min() < 0 and df[col].dropna().max() > 0:
                            log2fc_col = col
                            break
                
                if not gene_col:
                    # Gene column typically has string values
                    string_cols = df.select_dtypes(include=['object']).columns.tolist()
                    if string_cols:
                        gene_col = string_cols[0]
            
            if not all([p_val_col, log2fc_col, gene_col]):
                raise ValueError("Could not identify required columns in the DE results file")
            
            # Filter for significant genes
            significant_df = df[
                (df[p_val_col] <= p_value_threshold) & 
                (abs(df[log2fc_col]) >= log2fc_threshold)
            ].sort_values(by=p_val_col)
            
            # Get top upregulated and downregulated genes
            upregulated = significant_df[significant_df[log2fc_col] > 0].head(max_genes // 2)
            downregulated = significant_df[significant_df[log2fc_col] < 0].head(max_genes // 2)
            
            # Combine and get the gene list
            top_genes = pd.concat([upregulated, downregulated])
            gene_list = top_genes[gene_col].tolist()
            
            # Create result dictionary
            result = {
                'significant_genes': gene_list,
                'upregulated_genes': upregulated[gene_col].tolist(),
                'downregulated_genes': downregulated[gene_col].tolist(),
                'total_significant': len(significant_df),
                'selected_genes': len(gene_list)
            }
            
            return result
            
        except Exception as e:
            print(f"Error parsing DE results: {e}")
            return {'error': str(e)}
    
    def parse_gsea_results(self, file_path, format_type='csv', p_value_threshold=0.05, 
                          max_pathways=30):
        """
        Parse gene set enrichment analysis results file.
        
        Args:
            file_path (str): Path to GSEA results file
            format_type (str): File format ('csv', 'tsv', 'excel')
            p_value_threshold (float): P-value significance threshold
            max_pathways (int): Maximum number of pathways to include
            
        Returns:
            dict: Parsed GSEA results with significant pathways
        """
        try:
            # Load the file based on format
            if format_type.lower() == 'csv':
                df = pd.read_csv(file_path)
            elif format_type.lower() == 'tsv':
                df = pd.read_csv(file_path, sep='\t')
            elif format_type.lower() == 'excel':
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            # Detect column names for p-value and pathway/term
            # Common column names for p-value
            p_value_cols = ['p_val', 'p_value', 'pvalue', 'p.value', 'p-value', 'p_val_adj', 
                           'p_adj', 'padj', 'adj.p.val', 'adj_p_val', 'fdr', 'q_value']
            
            # Common column names for pathway/term
            pathway_cols = ['pathway', 'term', 'pathway_name', 'term_name', 'description', 
                           'pathway_id', 'term_id', 'go_term', 'kegg_pathway']
            
            # Common column names for genes in pathway
            genes_cols = ['genes', 'gene_list', 'leading_edge', 'core_enrichment', 'overlap']
            
            # Find the actual column names in the dataframe
            p_val_col = next((col for col in p_value_cols if col.lower() in [c.lower() for c in df.columns]), None)
            pathway_col = next((col for col in pathway_cols if col.lower() in [c.lower() for c in df.columns]), None)
            genes_col = next((col for col in genes_cols if col.lower() in [c.lower() for c in df.columns]), None)
            
            if not all([p_val_col, pathway_col]):
                # If automatic detection fails, try to infer from column data types
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                string_cols = df.select_dtypes(include=['object']).columns.tolist()
                
                if not p_val_col and len(numeric_cols) >= 1:
                    # P-values are typically between 0 and 1
                    for col in numeric_cols:
                        if df[col].dropna().between(0, 1).all():
                            p_val_col = col
                            break
                
                if not pathway_col and len(string_cols) >= 1:
                    # Pathway column typically has the longest string values
                    avg_lengths = {col: df[col].astype(str).str.len().mean() for col in string_cols}
                    pathway_col = max(avg_lengths.items(), key=lambda x: x[1])[0]
                
                if not genes_col and len(string_cols) >= 2:
                    # Genes column often contains comma-separated lists
                    for col in string_cols:
                        if col != pathway_col and df[col].astype(str).str.contains(',').any():
                            genes_col = col
                            break
            
            if not all([p_val_col, pathway_col]):
                raise ValueError("Could not identify required columns in the GSEA results file")
            
            # Filter for significant pathways
            significant_df = df[df[p_val_col] <= p_value_threshold].sort_values(by=p_val_col)
            
            # Get top pathways
            top_pathways = significant_df.head(max_pathways)
            pathway_list = top_pathways[pathway_col].tolist()
            
            # Extract genes if available
            pathway_genes = {}
            if genes_col:
                for idx, row in top_pathways.iterrows():
                    pathway = row[pathway_col]
                    genes_str = str(row[genes_col])
                    # Handle different formats of gene lists
                    if ',' in genes_str:
                        genes = [g.strip() for g in genes_str.split(',')]
                    elif ';' in genes_str:
                        genes = [g.strip() for g in genes_str.split(';')]
                    elif ' ' in genes_str:
                        genes = [g.strip() for g in genes_str.split()]
                    else:
                        genes = [genes_str]
                    
                    pathway_genes[pathway] = genes
            
            # Create result dictionary
            result = {
                'significant_pathways': pathway_list,
                'pathway_genes': pathway_genes if genes_col else {},
                'total_significant': len(significant_df),
                'selected_pathways': len(pathway_list)
            }
            
            return result
            
        except Exception as e:
            print(f"Error parsing GSEA results: {e}")
            return {'error': str(e)}
    
    def extract_cell_types(self, file_path=None, cell_types=None):
        """
        Extract or set cell types for keyword generation.
        
        Args:
            file_path (str): Path to file with cell type information
            cell_types (list): Direct list of cell types
            
        Returns:
            list: Cell types for keyword generation
        """
        if cell_types is not None and isinstance(cell_types, list):
            return cell_types
        
        if file_path is not None:
            try:
                # Determine file type from extension
                ext = os.path.splitext(file_path)[1].lower()
                
                if ext in ['.csv', '.txt', '.tsv']:
                    # Try to read as CSV/TSV
                    sep = ',' if ext == '.csv' else '\t'
                    df = pd.read_csv(file_path, sep=sep)
                    
                    # Look for column with cell type information
                    cell_type_cols = ['cell_type', 'celltype', 'cell type', 'cluster', 'annotation']
                    cell_col = next((col for col in cell_type_cols if col.lower() in 
                                    [c.lower() for c in df.columns]), None)
                    
                    if cell_col:
                        return df[cell_col].unique().tolist()
                    else:
                        # If no specific column found, use the first column
                        return df.iloc[:, 0].unique().tolist()
                
                elif ext == '.json':
                    # Try to read as JSON
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict) and 'cell_types' in data:
                        return data['cell_types']
                    else:
                        # Try to extract from any field that might contain cell types
                        for key, value in data.items():
                            if 'cell' in key.lower() and isinstance(value, list):
                                return value
                
                # If all else fails, read as text and assume one cell type per line
                with open(file_path, 'r') as f:
                    return [line.strip() for line in f if line.strip()]
                
            except Exception as e:
                print(f"Error extracting cell types from file: {e}")
                return []
        
        # Default cell types if nothing provided
        return ['fibroblast', 'keratinocyte', 'endothelial cell', 'immune cell', 
                'melanocyte', 'stem cell']
    
    def generate_keywords(self, biological_system, condition, species, 
                         de_results=None, gsea_results=None, cell_types=None,
                         max_keywords_per_category=5, max_total_keywords=50):
        """
        Generate search keywords based on biological context and analysis results.
        
        Args:
            biological_system (str): Biological system (e.g., 'skin', 'liver')
            condition (str): Condition of interest (e.g., 'keloid', 'fibrosis')
            species (str): Species (e.g., 'human', 'mouse')
            de_results (dict): Differential expression results
            gsea_results (dict): GSEA results
            cell_types (list): Cell types of interest
            max_keywords_per_category (int): Maximum keywords per category
            max_total_keywords (int): Maximum total keywords to generate
            
        Returns:
            dict: Generated keywords by category and combined queries
        """
        keywords = {
            'context': [],
            'genes': [],
            'pathways': [],
            'cell_types': [],
            'combined_queries': []
        }
        
        # Add context keywords
        context_terms = [
            f"{species} {biological_system}",
            f"{species} {condition}",
            f"{biological_system} {condition}",
            f"{species} {biological_system} {condition}",
            "single cell RNA-seq",
            "scRNA-seq",
            "transcriptomics",
            "differential expression",
            "gene expression"
        ]
        keywords['context'] = context_terms[:max_keywords_per_category]
        
        # Add cell type keywords
        if cell_types:
            cell_type_terms = []
            for cell in cell_types[:max_keywords_per_category]:
                cell_type_terms.append(f"{cell} {condition}")
                cell_type_terms.append(f"{cell} {biological_system}")
                cell_type_terms.append(f"{species} {cell} {condition}")
            
            keywords['cell_types'] = cell_type_terms[:max_keywords_per_category]
        
        # Add gene keywords from DE results
        if de_results and 'significant_genes' in de_results:
            gene_terms = []
            for gene in de_results['significant_genes'][:max_keywords_per_category]:
                gene_terms.append(f"{gene} {condition}")
                gene_terms.append(f"{gene} {biological_system}")
                gene_terms.append(f"{gene} {species} {condition}")
            
            keywords['genes'] = gene_terms[:max_keywords_per_category]
        
        # Add pathway keywords from GSEA results
        if gsea_results and 'significant_pathways' in gsea_results:
            pathway_terms = []
            for pathway in gsea_results['significant_pathways'][:max_keywords_per_category]:
                # Clean pathway name for better search results
                clean_pathway = pathway.replace('_', ' ').replace('KEGG_', '').replace('GO_', '')
                pathway_terms.append(f"{clean_pathway} {condition}")
                pathway_terms.append(f"{clean_pathway} {biological_system}")
                pathway_terms.append(f"{clean_pathway} {species}")
            
            keywords['pathways'] = pathway_terms[:max_keywords_per_category]
        
        # Generate combined queries
        combined_queries = []
        
        # Core biological context
        combined_queries.append(f"{species} {biological_system} {condition} gene expression")
        combined_queries.append(f"{species} {biological_system} {condition} transcriptome")
        combined_queries.append(f"{species} {biological_system} {condition} single cell")
        
        # Add cell type specific queries
        if cell_types:
            for cell in cell_types[:3]:  # Limit to top 3 cell types
                combined_queries.append(f"{species} {cell} {condition} gene expression")
                combined_queries.append(f"{species} {cell} {biological_system} {condition}")
        
        # Add gene specific queries
        if de_results and 'significant_genes' in de_results:
            # Top upregulated genes
            if 'upregulated_genes' in de_results and de_results['upregulated_genes']:
                top_up = de_results['upregulated_genes'][:3]  # Top 3 upregulated
                combined_queries.append(f"{species} {condition} {' '.join(top_up)}")
                combined_queries.append(f"{biological_system} {' '.join(top_up)}")
            
            # Top downregulated genes
            if 'downregulated_genes' in de_results and de_results['downregulated_genes']:
                top_down = de_results['downregulated_genes'][:3]  # Top 3 downregulated
                combined_queries.append(f"{species} {condition} {' '.join(top_down)}")
                combined_queries.append(f"{biological_system} {' '.join(top_down)}")
        
        # Add pathway specific queries
        if gsea_results and 'significant_pathways' in gsea_results:
            top_pathways = gsea_results['significant_pathways'][:3]  # Top 3 pathways
            for pathway in top_pathways:
                clean_pathway = pathway.replace('_', ' ').replace('KEGG_', '').replace('GO_', '')
                combined_queries.append(f"{species} {condition} {clean_pathway}")
                combined_queries.append(f"{biological_system} {clean_pathway}")
        
        # Limit to max total keywords
        keywords['combined_queries'] = combined_queries[:max_total_keywords]
        
        # Save generated keywords
        self._save_keywords(keywords, biological_system, condition, species)
        
        return keywords
    
    def _save_keywords(self, keywords, biological_system, condition, species):
        """Save generated keywords to file."""
        filename = f"{species}_{biological_system}_{condition}_keywords.json"
        filepath = os.path.join(self.output_dir, filename.replace(' ', '_').lower())
        
        with open(filepath, 'w') as f:
            json.dump(keywords, f, indent=2)
        
        print(f"Keywords saved to {filepath}")
        
        # Also save a text file with just the combined queries for easy use
        queries_file = os.path.splitext(filepath)[0] + "_queries.txt"
        with open(queries_file, 'w') as f:
            for query in keywords['combined_queries']:
                f.write(f"{query}\n")
        
        print(f"Search queries saved to {queries_file}")
    
    def generate_iterative_gene_queries(self, genes, biological_system, condition, species, 
                                       batch_size=3, max_queries=50):
        """
        Generate iterative queries for a list of genes.
        
        Args:
            genes (list): List of gene symbols
            biological_system (str): Biological system
            condition (str): Condition of interest
            species (str): Species
            batch_size (int): Number of genes per query
            max_queries (int): Maximum number of queries to generate
            
        Returns:
            list: List of search queries
        """
        if not genes:
            return []
        
        queries = []
        
        # Base context
        base_context = f"{species} {biological_system} {condition}"
        
        # Process genes in batches
        for i in range(0, len(genes), batch_size):
            if len(queries) >= max_queries:
                break
                
            gene_batch = genes[i:i+batch_size]
            gene_str = ' '.join(gene_batch)
            
            # Create queries with this gene batch
            queries.append(f"{base_context} {gene_str}")
            queries.append(f"{species} {gene_str} expression")
            queries.append(f"{condition} {gene_str} expression")
        
        return queries[:max_queries]
    
    def generate_iterative_pathway_queries(self, pathways, biological_system, condition, species,
                                         max_queries=50):
        """
        Generate iterative queries for a list of pathways.
        
        Args:
            pathways (list): List of pathway names
            biological_system (str): Biological system
            condition (str): Condition of interest
            species (str): Species
            max_queries (int): Maximum number of queries to generate
            
        Returns:
            list: List of search queries
        """
        if not pathways:
            return []
        
        queries = []
        
        # Base context
        base_context = f"{species} {biological_system} {condition}"
        
        # Process each pathway individually
        for pathway in pathways:
            if len(queries) >= max_queries:
                break
                
            # Clean pathway name
            clean_pathway = pathway.replace('_', ' ').replace('KEGG_', '').replace('GO_', '')
            
            # Create queries with this pathway
            queries.append(f"{base_context} {clean_pathway}")
            queries.append(f"{species} {clean_pathway}")
            queries.append(f"{condition} {clean_pathway}")
        
        return queries[:max_queries]

# Example usage
if __name__ == "__main__":
    # Initialize the keyword generator
    kg = KeywordGenerator(output_dir="generated_keywords")
    
    # Example with mock data
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
    keywords = kg.generate_keywords(
        biological_system='skin',
        condition='keloid',
        species='human',
        de_results=de_results,
        gsea_results=gsea_results,
        cell_types=cell_types
    )
    
    print(json.dumps(keywords, indent=2))
