# modules/notifier.py  — v3.0
"""
CHANGES FROM v2.0:
  CHANGE: Alert emails now sent FROM phishgaurdai@gmail.com
          TO the monitored user (EMAIL_ADDRESS / ALERT_RECIPIENT_EMAIL).
          Uses ALERT_SENDER_EMAIL + ALERT_SENDER_PASSWORD from .env.
          Self-email loop fix no longer needed since sender and
          recipient are now different accounts — but kept as safety net.
"""

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header
from typing import Dict
from config.settings import config

logger = logging.getLogger(__name__)


def _decode_subject(subject: str) -> str:
    """Decode RFC 2047 encoded subject like =?utf-8?q?...? to plain text."""
    try:
        parts = decode_header(subject)
        decoded = ""
        for part, enc in parts:
            if isinstance(part, bytes):
                decoded += part.decode(enc or "utf-8", errors="replace")
            else:
                decoded += str(part)
        return decoded
    except Exception:
        return subject


class PhishingNotifier:
    """
    Sends alerts when PhishGuard detects a phishing email.

    Flow:
      phishgaurdai@gmail.com  →  smartnaniallu007@gmail.com
      (ALERT_SENDER_EMAIL)        (ALERT_RECIPIENT_EMAIL)
    """

    def notify(self, result: Dict, parsed_email: Dict) -> None:
        if not result.get("is_phishing"):
            return

        sender = parsed_email.get("headers", {}).get("from", "")

        # Safety net: skip if somehow our own alert email gets re-scanned
        if config.ALERT_SENDER_EMAIL.lower() in sender.lower():
            logger.debug("[Notifier] Skipping PhishGuard alert email")
            return

        raw_subject = parsed_email.get("headers", {}).get("subject", "Unknown Subject")
        subject     = _decode_subject(raw_subject)
        confidence  = result.get("confidence", 0)
        risk_score  = result.get("risk_score", 0)
        signals     = result.get("key_signals", [])
        explanation = result.get("explanation", "")

        self._toast(subject, sender, confidence)
        self._gmail(subject, sender, confidence, risk_score, signals, explanation)

    def _toast(self, subject: str, sender: str, confidence: float) -> None:
        """Windows toast via plyer."""
        try:
            from plyer import notification
            notification.notify(
                title="PhishGuard: Phishing Detected",
                message=(
                    f"From: {sender[:60]}\n"
                    f"Subject: {subject[:80]}\n"
                    f"Confidence: {confidence:.1%}"
                ),
                app_name="PhishGuard AI",
                timeout=10,
            )
            logger.info(f"[Notifier] Toast sent for: {subject[:50]}")
        except ImportError:
            logger.warning("[Notifier] plyer not installed. Run: pip install plyer")
        except Exception as e:
            logger.error(f"[Notifier] Toast failed: {e}")

    def _gmail(
        self,
        subject: str,
        sender: str,
        confidence: float,
        risk_score: float,
        signals: list,
        explanation: str,
    ) -> None:
        """
        Send alert FROM phishgaurdai@gmail.com TO smartnaniallu007@gmail.com.
        Uses ALERT_SENDER_EMAIL + ALERT_SENDER_PASSWORD from .env.
        """
        if not config.ALERT_SENDER_PASSWORD:
            logger.warning(
                "[Notifier] ALERT_SENDER_PASSWORD not set in .env — "
                "skipping Gmail alert. Add App Password for phishgaurdai@gmail.com."
            )
            return

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[PhishGuard Alert] Phishing detected — {subject[:60]}"
            msg["From"]    = f"PhishGuard AI <{config.ALERT_SENDER_EMAIL}>"
            msg["To"]      = config.ALERT_RECIPIENT_EMAIL

            signals_html = "".join(f"<li>{s}</li>" for s in signals) or "<li>None detected</li>"

            html_body = f"""
            <html><body style="font-family:Arial,sans-serif;max-width:600px;">
              <div style="background:#c0392b;color:white;padding:16px;border-radius:6px 6px 0 0;">
                <h2 style="margin:0;">PhishGuard Alert</h2>
                <p style="margin:4px 0 0;">Phishing email detected in your inbox</p>
              </div>
              <div style="border:1px solid #ddd;border-top:none;padding:20px;border-radius:0 0 6px 6px;">
                <table style="width:100%;border-collapse:collapse;">
                  <tr><td style="color:#666;padding:6px 0;width:140px;"><b>From</b></td>
                      <td style="padding:6px 0;">{sender}</td></tr>
                  <tr><td style="color:#666;padding:6px 0;"><b>Subject</b></td>
                      <td style="padding:6px 0;">{subject}</td></tr>
                  <tr><td style="color:#666;padding:6px 0;"><b>Confidence</b></td>
                      <td style="padding:6px 0;">{confidence:.1%}</td></tr>
                  <tr><td style="color:#666;padding:6px 0;"><b>Risk Score</b></td>
                      <td style="padding:6px 0;">{risk_score:.0f} / 100</td></tr>
                </table>
                <h3 style="margin-top:20px;color:#c0392b;">Signals Detected</h3>
                <ul style="margin:0;padding-left:20px;color:#333;">{signals_html}</ul>
                <h3 style="margin-top:20px;color:#333;">Explanation</h3>
                <p style="color:#555;background:#f9f9f9;padding:12px;border-radius:4px;">{explanation}</p>
                <p style="margin-top:20px;font-size:12px;color:#999;">
                  Do not click any links or open attachments in the flagged email.<br>
                  Sent by PhishGuard AI v5.0 — phishgaurdai@gmail.com
                </p>
              </div>
            </body></html>
            """

            plain_body = (
                f"PHISHING DETECTED\n\n"
                f"From     : {sender}\n"
                f"Subject  : {subject}\n"
                f"Confidence: {confidence:.1%}\n"
                f"Risk Score: {risk_score:.0f}/100\n\n"
                f"Signals:\n" + "\n".join(f"  - {s}" for s in signals) + "\n\n"
                f"Explanation:\n{explanation}\n\n"
                f"Do not click any links or open attachments in the flagged email."
            )

            msg.attach(MIMEText(plain_body, "plain"))
            msg.attach(MIMEText(html_body,  "html"))

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(config.ALERT_SENDER_EMAIL, config.ALERT_SENDER_PASSWORD)
                server.sendmail(
                    config.ALERT_SENDER_EMAIL,
                    config.ALERT_RECIPIENT_EMAIL,
                    msg.as_string(),
                )

            logger.info(
                f"[Notifier] Alert sent: {config.ALERT_SENDER_EMAIL} → "
                f"{config.ALERT_RECIPIENT_EMAIL} | {subject[:40]}"
            )

        except smtplib.SMTPAuthenticationError:
            logger.error(
                "[Notifier] Gmail auth failed for phishgaurdai@gmail.com. "
                "Check ALERT_SENDER_PASSWORD in .env — must be an App Password."
            )
        except Exception as e:
            logger.error(f"[Notifier] Gmail alert failed: {e}")