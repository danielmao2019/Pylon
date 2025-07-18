from typing import List, Dict, Any, Optional
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime


class EmailSender:
    """Handles email sending for daily summaries and notifications."""
    
    def __init__(self, 
                 smtp_host: str,
                 smtp_port: int,
                 username: str,
                 password: str,
                 from_addr: str,
                 to_addrs: List[str],
                 enabled: bool = True):
        """Initialize email sender.
        
        Args:
            smtp_host: SMTP server hostname
            smtp_port: SMTP server port
            username: Email username for authentication
            password: Email password for authentication
            from_addr: From email address
            to_addrs: List of recipient email addresses
            enabled: Whether email sending is enabled
        """
        assert isinstance(smtp_host, str), f"smtp_host must be str, got {type(smtp_host)}"
        assert isinstance(smtp_port, int), f"smtp_port must be int, got {type(smtp_port)}"
        assert isinstance(username, str), f"username must be str, got {type(username)}"
        assert isinstance(password, str), f"password must be str, got {type(password)}"
        assert isinstance(from_addr, str), f"from_addr must be str, got {type(from_addr)}"
        assert isinstance(to_addrs, list), f"to_addrs must be list, got {type(to_addrs)}"
        assert isinstance(enabled, bool), f"enabled must be bool, got {type(enabled)}"
        
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs
        self.enabled = enabled
    
    def send_daily_summary(self, summary_content: str, date: Optional[str] = None) -> bool:
        """Send daily summary via email.
        
        Args:
            summary_content: The markdown summary content to send
            date: Date string for the summary (defaults to today)
            
        Returns:
            True if email sent successfully, False otherwise
        """
        if not self.enabled:
            print("Email sending is disabled")
            return False
        
        if not self._validate_config():
            print("Email configuration is incomplete")
            return False
        
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            # Create message
            msg = self._create_summary_message(summary_content, date)
            
            # Send email
            success = self._send_message(msg)
            
            if success:
                print(f"✅ Daily summary email sent successfully to {len(self.to_addrs)} recipient(s)")
            else:
                print("❌ Failed to send daily summary email")
            
            return success
            
        except Exception as e:
            print(f"❌ Error sending daily summary email: {e}")
            return False
    
    def _validate_config(self) -> bool:
        """Validate email configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        if not self.smtp_host or not self.smtp_port:
            print("Missing SMTP host or port configuration")
            return False
        
        if not self.username or not self.password:
            print("Missing email username or password")
            return False
        
        if not self.from_addr:
            print("Missing from email address")
            return False
        
        if not self.to_addrs:
            print("Missing recipient email addresses")
            return False
        
        return True
    
    def _create_summary_message(self, summary_content: str, date: str) -> MIMEMultipart:
        """Create email message for daily summary.
        
        Args:
            summary_content: The markdown summary content
            date: Date string for the summary
            
        Returns:
            Email message object
        """
        msg = MIMEMultipart('alternative')
        
        # Email headers
        msg['Subject'] = f"Pylon Daily Summary - {date}"
        msg['From'] = self.from_addr
        msg['To'] = ', '.join(self.to_addrs)
        
        # Create both plain text and HTML versions
        text_content = self._convert_markdown_to_text(summary_content)
        html_content = self._convert_markdown_to_html(summary_content)
        
        # Attach both versions
        part1 = MIMEText(text_content, 'plain')
        part2 = MIMEText(html_content, 'html')
        
        msg.attach(part1)
        msg.attach(part2)
        
        return msg
    
    def _convert_markdown_to_text(self, markdown_content: str) -> str:
        """Convert markdown to plain text for email.
        
        Args:
            markdown_content: Markdown content to convert
            
        Returns:
            Plain text version of the content
        """
        # Simple markdown to text conversion
        text_content = markdown_content
        
        # Remove markdown headers
        text_content = text_content.replace('# ', '')
        text_content = text_content.replace('## ', '')
        text_content = text_content.replace('### ', '')
        
        # Remove markdown bold
        text_content = text_content.replace('**', '')
        
        # Clean up extra whitespace
        lines = text_content.split('\n')
        cleaned_lines = []
        for line in lines:
            cleaned_lines.append(line.rstrip())
        
        return '\n'.join(cleaned_lines)
    
    def _convert_markdown_to_html(self, markdown_content: str) -> str:
        """Convert markdown to HTML for email.
        
        Args:
            markdown_content: Markdown content to convert
            
        Returns:
            HTML version of the content
        """
        # Simple markdown to HTML conversion
        html_content = markdown_content
        
        # Convert headers
        html_content = html_content.replace('# ', '<h1>').replace('\n## ', '</h1>\n<h2>')
        html_content = html_content.replace('## ', '<h2>').replace('\n### ', '</h2>\n<h3>')
        html_content = html_content.replace('### ', '<h3>')
        
        # Convert bold text
        import re
        html_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html_content)
        
        # Convert line breaks to HTML
        html_content = html_content.replace('\n', '<br>\n')
        
        # Add proper HTML structure
        html_template = f"""
        <html>
          <head>
            <style>
              body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
              h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
              h2 {{ color: #34495e; margin-top: 25px; }}
              h3 {{ color: #7f8c8d; }}
              strong {{ color: #2c3e50; }}
              pre {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; }}
            </style>
          </head>
          <body>
            {html_content}
          </body>
        </html>
        """
        
        return html_template
    
    def _send_message(self, msg: MIMEMultipart) -> bool:
        """Send email message via SMTP.
        
        Args:
            msg: Email message to send
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Create SSL context
            context = ssl.create_default_context()
            
            # Connect to server and send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.username, self.password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())
            
            return True
            
        except smtplib.SMTPException as e:
            print(f"SMTP error: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error sending email: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test email server connection and authentication.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not self._validate_config():
            return False
        
        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.username, self.password)
            
            print("✅ Email server connection test successful")
            return True
            
        except Exception as e:
            print(f"❌ Email server connection test failed: {e}")
            return False