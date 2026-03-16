# modules/email_fetcher.py
import imaplib
import email
import json
from typing import List, Dict
from config.settings import config


class EmailFetcher:
    """
    Connects to IMAP server and fetches raw emails.
    Returns list of raw email.message.Message objects.
    
    Why IMAP? It's the universal protocol. Works with Gmail,
    Outlook, ProtonMail bridge, corporate Exchange servers.
    """
    
    def __init__(self):
        self.connection = None
    
    def connect(self) -> bool:
        """Establish SSL IMAP connection."""
        try:
            self.connection = imaplib.IMAP4_SSL(
                config.IMAP_SERVER, 
                config.IMAP_PORT
            )
            self.connection.login(
                config.EMAIL_ADDRESS, 
                config.EMAIL_PASSWORD
            )
            print(f"[+] Connected to {config.IMAP_SERVER}")
            return True
        except imaplib.IMAP4.error as e:
            print(f"[-] IMAP connection failed: {e}")
            return False
    
    def fetch_emails(self, folder: str = "INBOX", count: int = 5) -> List[email.message.Message]:
        """
        Fetch N most recent emails from folder.
        
        Args:
            folder: Mailbox folder (INBOX, SPAM, etc.)
            count: Number of emails to fetch
            
        Returns:
            List of raw email Message objects
        """
        if not self.connection:
            raise ConnectionError("Not connected. Call connect() first.")
        
        self.connection.select(folder)
        
        # Search all emails, get IDs
        _, message_ids = self.connection.search(None, "ALL")
        email_ids = message_ids[0].split()
        
        # Take the last N (most recent)
        recent_ids = email_ids[-count:] if len(email_ids) >= count else email_ids
        
        raw_emails = []
        for email_id in recent_ids:
            _, msg_data = self.connection.fetch(email_id, "(RFC822)")
            raw_email = email.message_from_bytes(msg_data[0][1])
            raw_emails.append(raw_email)
        
        print(f"[+] Fetched {len(raw_emails)} emails from {folder}")
        return raw_emails
    
    def fetch_spam_folder(self, count: int = 50) -> List[email.message.Message]:
        """
        Specifically target spam/junk folder — 
        this is where real-world phishing lives.
        """
        # Gmail uses [Gmail]/Spam
        spam_folders = ["[Gmail]/Spam", "Junk", "SPAM", "Spam"]
        
        for folder in spam_folders:
            try:
                return self.fetch_emails(folder=folder, count=count)
            except Exception:
                continue
        
        print("[-] No spam folder found, trying INBOX")
        return self.fetch_emails(count=count)
    
    def disconnect(self):
        if self.connection:
            self.connection.logout()
            print("[+] Disconnected from IMAP")