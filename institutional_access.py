#!/usr/bin/env python3
"""
Institutional Access Module for Paper Retrieval Pipeline

This module provides functionality to access paywalled articles using
institutional credentials (specifically for University of Michigan Ann Arbor).
It handles authentication with the university proxy and maintains a session
for downloading papers that would otherwise be behind paywalls.
"""

import os
import json
import time
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

class InstitutionalAccess:
    """
    Handles institutional access to paywalled articles using university credentials.
    
    This class provides methods to authenticate with a university proxy server
    and maintain a session for downloading papers that would otherwise be
    behind paywalls. It specifically supports University of Michigan Ann Arbor
    authentication methods.
    """
    
    def __init__(self, credentials_file=None, log_level=logging.INFO):
        """
        Initialize the institutional access handler.
        
        Args:
            credentials_file (str): Path to credentials JSON file
            log_level (int): Logging level
        """
        self.logger = self._setup_logger(log_level)
        self.session = requests.Session()
        self.authenticated = False
        self.last_auth_time = 0
        self.auth_valid_duration = 3600  # 1 hour in seconds
        
        # Set default headers to mimic a browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        # Load credentials
        self.credentials = self._load_credentials(credentials_file)
        
        # University of Michigan specific settings
        self.umich_proxy_url = "https://proxy.lib.umich.edu/login"
        self.umich_login_url = "https://weblogin.umich.edu/"
        
    def _setup_logger(self, log_level):
        """Set up logger for the institutional access module."""
        logger = logging.getLogger('institutional_access')
        logger.setLevel(log_level)
        
        # Create console handler
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_credentials(self, credentials_file):
        """
        Load credentials from a JSON file.
        
        Args:
            credentials_file (str): Path to credentials JSON file
            
        Returns:
            dict: Credentials dictionary
        """
        if not credentials_file:
            credentials_file = 'credentials.json'
        
        credentials = {}
        
        if os.path.exists(credentials_file):
            try:
                with open(credentials_file, 'r') as f:
                    credentials = json.load(f)
                self.logger.info(f"Loaded credentials from {credentials_file}")
            except Exception as e:
                self.logger.error(f"Error loading credentials: {e}")
        else:
            self.logger.warning(f"Credentials file {credentials_file} not found")
            # Create a template credentials file
            template = {
                "username": "your_umich_username",
                "password": "your_umich_password",
                "proxy_url": self.umich_proxy_url
            }
            try:
                with open(credentials_file, 'w') as f:
                    json.dump(template, f, indent=2)
                self.logger.info(f"Created template credentials file at {credentials_file}")
                self.logger.info("Please edit this file with your actual credentials")
            except Exception as e:
                self.logger.error(f"Error creating template credentials file: {e}")
        
        return credentials
    
    def is_authenticated(self):
        """
        Check if the session is currently authenticated.
        
        Returns:
            bool: True if authenticated and session is still valid
        """
        current_time = time.time()
        if self.authenticated and (current_time - self.last_auth_time) < self.auth_valid_duration:
            return True
        return False
    
    def authenticate(self):
        """
        Authenticate with the University of Michigan proxy server.
        
        This method handles the authentication process for University of Michigan,
        which typically involves a multi-step process through their weblogin system.
        
        Returns:
            bool: True if authentication was successful
        """
        if self.is_authenticated():
            self.logger.info("Already authenticated")
            return True
        
        if not self.credentials or 'username' not in self.credentials or 'password' not in self.credentials:
            self.logger.error("Missing credentials. Please check your credentials.json file")
            return False
        
        username = self.credentials.get('username')
        password = self.credentials.get('password')
        proxy_url = self.credentials.get('proxy_url', self.umich_proxy_url)
        
        self.logger.info(f"Authenticating with {proxy_url}")
        
        try:
            # Step 1: Access the proxy login page
            response = self.session.get(proxy_url)
            
            if response.status_code != 200:
                self.logger.error(f"Failed to access proxy login page: {response.status_code}")
                return False
            
            # Step 2: Parse the login form to find the redirect to weblogin
            soup = BeautifulSoup(response.text, 'html.parser')
            login_form = soup.find('form')
            
            if not login_form:
                self.logger.error("Could not find login form")
                return False
            
            # Get the weblogin URL
            weblogin_url = login_form.get('action', '')
            if not weblogin_url.startswith('http'):
                weblogin_url = urljoin(self.umich_login_url, weblogin_url)
            
            # Get form fields
            form_data = {}
            for input_field in login_form.find_all('input'):
                if input_field.get('name'):
                    form_data[input_field.get('name')] = input_field.get('value', '')
            
            # Step 3: Submit to weblogin
            response = self.session.post(weblogin_url, data=form_data)
            
            if response.status_code != 200:
                self.logger.error(f"Failed to access weblogin: {response.status_code}")
                return False
            
            # Step 4: Parse the actual login form
            soup = BeautifulSoup(response.text, 'html.parser')
            login_form = soup.find('form', {'id': 'loginForm'})
            
            if not login_form:
                self.logger.error("Could not find weblogin form")
                return False
            
            # Get the login submission URL
            login_submit_url = login_form.get('action', '')
            if not login_submit_url.startswith('http'):
                login_submit_url = urljoin(weblogin_url, login_submit_url)
            
            # Prepare login data
            login_data = {}
            for input_field in login_form.find_all('input'):
                if input_field.get('name'):
                    login_data[input_field.get('name')] = input_field.get('value', '')
            
            # Add credentials
            login_data['login'] = username
            login_data['password'] = password
            
            # Step 5: Submit login credentials
            response = self.session.post(login_submit_url, data=login_data)
            
            if response.status_code != 200:
                self.logger.error(f"Login submission failed: {response.status_code}")
                return False
            
            # Step 6: Check for successful login
            if "Incorrect Login" in response.text or "Authentication failed" in response.text:
                self.logger.error("Authentication failed: Incorrect credentials")
                return False
            
            # Step 7: Follow redirects to complete authentication
            soup = BeautifulSoup(response.text, 'html.parser')
            redirect_form = soup.find('form')
            
            if redirect_form:
                redirect_url = redirect_form.get('action', '')
                redirect_data = {}
                
                for input_field in redirect_form.find_all('input'):
                    if input_field.get('name'):
                        redirect_data[input_field.get('name')] = input_field.get('value', '')
                
                response = self.session.post(redirect_url, data=redirect_data)
                
                if response.status_code != 200:
                    self.logger.error(f"Redirect after login failed: {response.status_code}")
                    return False
            
            # Authentication successful
            self.authenticated = True
            self.last_auth_time = time.time()
            self.logger.info("Authentication successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return False
    
    def get_proxied_url(self, original_url):
        """
        Convert a regular URL to a proxied URL for institutional access.
        
        Args:
            original_url (str): Original publisher URL
            
        Returns:
            str: Proxied URL for institutional access
        """
        if not original_url:
            return None
        
        # Parse the original URL
        parsed_url = urlparse(original_url)
        
        # University of Michigan proxy format
        proxied_url = f"https://proxy.lib.umich.edu/login?url={original_url}"
        
        return proxied_url
    
    def download_paper(self, url, output_path):
        """
        Download a paper using institutional access.
        
        Args:
            url (str): URL of the paper
            output_path (str): Path to save the downloaded paper
            
        Returns:
            bool: True if download was successful
        """
        if not url:
            self.logger.error("No URL provided")
            return False
        
        # Authenticate if needed
        if not self.is_authenticated():
            if not self.authenticate():
                self.logger.error("Authentication failed, cannot download paper")
                return False
        
        # Get proxied URL
        proxied_url = self.get_proxied_url(url)
        
        try:
            self.logger.info(f"Downloading paper from {proxied_url}")
            
            # Download the paper
            response = self.session.get(proxied_url, stream=True)
            
            if response.status_code != 200:
                self.logger.error(f"Download failed: {response.status_code}")
                return False
            
            # Check content type
            content_type = response.headers.get('Content-Type', '')
            if 'pdf' not in content_type.lower() and 'application/octet-stream' not in content_type.lower():
                # This might be an intermediate page, try to find the PDF link
                soup = BeautifulSoup(response.text, 'html.parser')
                pdf_link = None
                
                # Look for PDF links
                for link in soup.find_all('a'):
                    href = link.get('href', '')
                    if 'pdf' in href.lower() or (link.text and 'pdf' in link.text.lower()):
                        pdf_link = href
                        break
                
                if pdf_link:
                    # Make the PDF link absolute
                    if not pdf_link.startswith('http'):
                        pdf_link = urljoin(response.url, pdf_link)
                    
                    # Download the actual PDF
                    self.logger.info(f"Found PDF link: {pdf_link}")
                    response = self.session.get(pdf_link, stream=True)
                    
                    if response.status_code != 200:
                        self.logger.error(f"PDF download failed: {response.status_code}")
                        return False
                else:
                    self.logger.error("Could not find PDF link on the page")
                    return False
            
            # Save the PDF
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.logger.info(f"Paper downloaded successfully to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Download error: {e}")
            return False
    
    def test_authentication(self):
        """
        Test the authentication process.
        
        Returns:
            bool: True if authentication test was successful
        """
        if not self.authenticate():
            return False
        
        # Test access to a known paywalled article
        test_url = "https://www.sciencedirect.com/science/article/abs/pii/S0022202X15370834"
        proxied_url = self.get_proxied_url(test_url)
        
        try:
            response = self.session.get(proxied_url)
            
            if response.status_code != 200:
                self.logger.error(f"Test access failed: {response.status_code}")
                return False
            
            # Check if we got the actual content and not a login page
            if "Sign in to access" in response.text or "Please sign in" in response.text:
                self.logger.error("Test access failed: Still hitting paywall")
                return False
            
            self.logger.info("Authentication test successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Test error: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize institutional access
    inst_access = InstitutionalAccess()
    
    # Test authentication
    if inst_access.test_authentication():
        print("Authentication successful!")
        
        # Example download
        paper_url = "https://www.sciencedirect.com/science/article/abs/pii/S0022202X15370834"
        output_path = "test_paper.pdf"
        
        if inst_access.download_paper(paper_url, output_path):
            print(f"Paper downloaded to {output_path}")
        else:
            print("Paper download failed")
    else:
        print("Authentication failed")
