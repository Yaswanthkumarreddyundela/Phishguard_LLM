# modules/email_parser.py
import email
import hashlib
import json
import re
from email import policy
from email.utils import parseaddr, parsedate_to_datetime
from bs4 import BeautifulSoup
from typing import Dict, Optional, List


class EmailParser:
    """
    Converts raw email.message.Message objects into structured JSON dicts.
    
    Why structured JSON?
    - Consistent schema across all downstream modules
    - Easy serialization to database
    - Testable without live IMAP connection
    - Can ingest emails from ANY source (files, APIs, webhooks)
    """
    
    def parse(self, raw_email: email.message.Message) -> Dict:
        """
        Master parse function. Returns fully structured email dict.
        
        Output Schema:
        {
            "email_hash": str,          # SHA256 of raw content (deduplication key)
            "headers": {
                "from": str,
                "to": str,
                "reply_to": str,
                "subject": str,
                "date": str,
                "message_id": str,
                "x_mailer": str,
                "received": list[str],
                "spf_result": str,
                "dkim_result": str,
                "dmarc_result": str
            },
            "body": {
                "plain_text": str,
                "html_text": str,
                "combined": str         # Used for NLP input
            },
            "urls": list[str],          # All extracted URLs
            "attachments": list[dict],  # Attachment metadata
            "metadata": {
                "has_html": bool,
                "has_attachments": bool,
                "url_count": int,
                "raw_size_bytes": int
            }
        }
        """
        
        parsed = {
            "email_hash": self._compute_hash(raw_email),
            "headers": self._parse_headers(raw_email),
            "body": self._parse_body(raw_email),
            "urls": [],
            "attachments": [],
            "metadata": {}
        }
        
        # Extract URLs from body
        parsed["urls"] = self._extract_urls(
            parsed["body"]["plain_text"] + " " + parsed["body"]["html_text"]
        )
        
        # Parse attachments
        parsed["attachments"] = self._parse_attachments(raw_email)
        
        # Build metadata
        parsed["metadata"] = {
            "has_html": bool(parsed["body"]["html_text"]),
            "has_attachments": len(parsed["attachments"]) > 0,
            "url_count": len(parsed["urls"]),
            "raw_size_bytes": len(str(raw_email))
        }
        
        return parsed
    
    def _compute_hash(self, raw_email: email.message.Message) -> str:
        """SHA256 hash for deduplication and database keying."""
        content = str(raw_email).encode("utf-8", errors="replace")
        return hashlib.sha256(content).hexdigest()
    
    def _parse_headers(self, raw_email: email.message.Message) -> Dict:
        """
        Extract all security-relevant headers.
        SPF/DKIM/DMARC are critical signals — phishing emails 
        often fail these authentication checks.
        """
        
        def get_header(name: str) -> str:
            val = raw_email.get(name, "")
            return str(val).strip() if val else ""
        
        # Extract all 'Received' headers (email routing chain)
        received = raw_email.get_all("Received", [])
        
        # Parse SPF/DKIM/DMARC from Authentication-Results header
        auth_results = get_header("Authentication-Results")
        spf = self._extract_auth_result(auth_results, "spf")
        dkim = self._extract_auth_result(auth_results, "dkim")
        dmarc = self._extract_auth_result(auth_results, "dmarc")
        
        return {
            "from": get_header("From"),
            "to": get_header("To"),
            "reply_to": get_header("Reply-To"),
            "subject": get_header("Subject"),
            "date": get_header("Date"),
            "message_id": get_header("Message-ID"),
            "x_mailer": get_header("X-Mailer"),
            "x_originating_ip": get_header("X-Originating-IP"),
            "received": [str(r) for r in received],
            "spf_result": spf,
            "dkim_result": dkim,
            "dmarc_result": dmarc,
            "content_type": get_header("Content-Type")
        }
    
    def _extract_auth_result(self, auth_header: str, protocol: str) -> str:
        """Extract pass/fail/none from Authentication-Results header."""
        if not auth_header:
            return "none"
        pattern = rf"{protocol}=(\w+)"
        match = re.search(pattern, auth_header, re.IGNORECASE)
        return match.group(1).lower() if match else "none"
    
    def _parse_body(self, raw_email: email.message.Message) -> Dict:
        """
        Extract plain text and HTML body.
        We keep both: plain text for NLP, HTML for URL extraction.
        """
        plain_text = ""
        html_text = ""
        
        if raw_email.is_multipart():
            for part in raw_email.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))
                
                # Skip attachments
                if "attachment" in content_disposition:
                    continue
                
                try:
                    payload = part.get_payload(decode=True)
                    if payload is None:
                        continue
                    charset = part.get_content_charset() or "utf-8"
                    decoded = payload.decode(charset, errors="replace")
                    
                    if content_type == "text/plain":
                        plain_text += decoded
                    elif content_type == "text/html":
                        html_text += decoded
                except Exception:
                    continue
        else:
            try:
                payload = raw_email.get_payload(decode=True)
                if payload:
                    charset = raw_email.get_content_charset() or "utf-8"
                    plain_text = payload.decode(charset, errors="replace")
            except Exception:
                pass
        
        # Extract readable text from HTML
        html_as_text = ""
        if html_text:
            try:
                soup = BeautifulSoup(html_text, "html.parser")
                html_as_text = soup.get_text(separator=" ", strip=True)
            except Exception:
                html_as_text = html_text
        
        # Combined = best text signal for NLP
        combined = (plain_text + " " + html_as_text).strip()
        
        return {
            "plain_text": plain_text[:5000],    # Cap at 5000 chars
            "html_text": html_text[:10000],
            "combined": combined[:5000]
        }
    
    def _extract_urls(self, text: str) -> List[str]:
        """
        Extract all URLs from text/HTML.
        Why this matters: phishing emails use disguised or 
        obfuscated URLs as the primary attack vector.
        """
        url_pattern = re.compile(
            r'https?://[^\s<>"{}|\\^`\[\]]+',
            re.IGNORECASE
        )
        urls = url_pattern.findall(text)
        # Deduplicate while preserving order
        seen = set()
        unique_urls = []
        for url in urls:
            clean_url = url.rstrip(".,;)")
            if clean_url not in seen:
                seen.add(clean_url)
                unique_urls.append(clean_url)
        return unique_urls
    
    def _parse_attachments(self, raw_email: email.message.Message) -> List[Dict]:
        """
        Extract attachment metadata (NOT content — security risk).
        Dangerous extensions are a key phishing signal.
        """
        dangerous_extensions = {
            ".exe", ".bat", ".cmd", ".js", ".vbs", ".scr",
            ".ps1", ".hta", ".jar", ".msi", ".iso"
        }
        
        attachments = []
        if raw_email.is_multipart():
            for part in raw_email.walk():
                disposition = str(part.get("Content-Disposition", ""))
                if "attachment" in disposition:
                    filename = part.get_filename() or "unknown"
                    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
                    attachments.append({
                        "filename": filename,
                        "content_type": part.get_content_type(),
                        "extension": ext,
                        "is_dangerous": ext in dangerous_extensions,
                        "size_bytes": len(part.get_payload() or "")
                    })
        return attachments
    
    def parse_from_file(self, filepath: str) -> Dict:
        """Parse a .eml file directly — useful for testing without IMAP."""
        with open(filepath, "rb") as f:
            raw_email = email.message_from_bytes(f.read())
        return self.parse(raw_email)
    
    def parse_from_string(self, raw_string: str) -> Dict:
        """Parse email from raw string — useful for unit tests."""
        raw_email = email.message_from_string(raw_string)
        return self.parse(raw_email)