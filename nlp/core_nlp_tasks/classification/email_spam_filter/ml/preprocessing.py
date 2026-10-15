"""Input normalization and demonstration data for spam classification."""

import re

import pandas as pd


SYNTHETIC_EMAILS = [
    ("ham", "Hi Maya, are we still meeting for lunch tomorrow at noon?"),
    ("ham", "The project review has moved to 10:30. I have shared the updated agenda."),
    ("ham", "Your library books are due next Friday. Renew them online if needed."),
    ("ham", "Thanks for sending the notes. I will review them before our call."),
    ("ham", "Your order has shipped and is expected to arrive on Tuesday."),
    ("ham", "Can you please send me the slides from yesterday's presentation?"),
    ("ham", "The monthly team newsletter is ready to read on the company intranet."),
    ("ham", "Reminder: your dentist appointment is scheduled for Thursday afternoon."),
    ("ham", "I made the changes you suggested and attached the revised document."),
    ("ham", "The train leaves at 6:15. Let me know when you get to the station."),
    ("ham", "Your electricity bill is available in your account. The due date is May 24."),
    ("ham", "Would you like to join us for a walk in the park this weekend?"),
    ("ham", "The workshop recording and course materials are now available to students."),
    ("ham", "I will be out of office on Monday and back on Tuesday morning."),
    ("ham", "Please confirm whether Friday works for the planning meeting."),
    ("ham", "Your password was changed successfully. Contact support if you did not make this change."),
    ("ham", "Here is the recipe I promised. Let me know how it turns out."),
    ("ham", "The package was left at the front desk for you to collect."),
    ("ham", "Our flight boards at gate 12. Boarding starts 40 minutes before departure."),
    ("ham", "I have booked a table for four at the Italian restaurant at seven."),
    ("spam", "URGENT: You have won a free gift card. Click here now to claim your prize!"),
    ("spam", "Congratulations, selected winner! Claim your cash reward today before it expires."),
    ("spam", "Act now to earn guaranteed income from home. Pay a small fee to get started."),
    ("spam", "Your account will be closed today. Verify your password at http://secure-account-check.example"),
    ("spam", "Final warning: pay the outstanding amount immediately to avoid legal action."),
    ("spam", "Exclusive deal! Buy now and get a free prize. Limited time offer, click to claim."),
    ("spam", "We detected unusual activity. Confirm your bank login and password immediately."),
    ("spam", "You have been selected for a cash bonus. Send your card details to receive payment."),
    ("spam", "Your package is on hold. Pay the delivery fee now using this unusual payment link."),
    ("spam", "Amazing weight loss results guaranteed. Order the miracle supplement today!"),
    ("spam", "Important account warning: your mailbox will be deleted unless you verify now."),
    ("spam", "Earn thousands each week with no experience. Reply with your personal details."),
    ("spam", "Your invoice is overdue. Wire the payment today or your service will be suspended."),
    ("spam", "Claim your exclusive reward now. This offer is only available for the next hour."),
    ("spam", "Security alert: log in to restore access and confirm your account credentials."),
    ("spam", "You are our lucky customer. Provide your banking details to release the prize money."),
    ("spam", "Low interest loan approved instantly. Pay the processing fee to unlock your funds."),
    ("spam", "Your subscription will renew for a high fee. Click this link to cancel immediately."),
    ("spam", "Urgent payment required today. Purchase gift cards and send us the redemption codes."),
    ("spam", "Free vacation waiting for you. Confirm your identity and claim your travel voucher."),
]


def normalize_text(text):
    """Return safe, consistent text for the vectorizer."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def load_synthetic_dataset():
    """Build the built-in balanced demonstration dataset."""
    return pd.DataFrame(SYNTHETIC_EMAILS, columns=["label", "text"])


def validate_dataset(frame):
    """Return a normalized two-column dataset, rejecting ambiguous inputs."""
    normalized_columns = {str(column).strip().lower(): column for column in frame.columns}
    label_column = next(
        (normalized_columns[name] for name in ("label", "category", "class", "target") if name in normalized_columns),
        None,
    )
    text_column = next(
        (normalized_columns[name] for name in ("text", "message", "email") if name in normalized_columns),
        None,
    )
    if label_column is None:
        raise ValueError("Dataset must include a label, category, class, or target column.")
    if text_column is None:
        subject_column = normalized_columns.get("subject")
        body_column = normalized_columns.get("body")
        if subject_column is None and body_column is None:
            raise ValueError("Dataset must include a text, body, message, email, or subject column.")
        text = (
            frame[subject_column].fillna("").astype(str)
            if body_column is None
            else frame[body_column].fillna("").astype(str)
        )
        if body_column is not None and subject_column is not None:
            text = frame[subject_column].fillna("").astype(str) + " " + text
    else:
        text = frame[text_column].fillna("").astype(str)

    labels = frame[label_column].fillna("").astype(str).str.strip().str.lower()
    if not labels.isin(["spam", "ham"]).all():
        invalid = sorted(set(labels) - {"spam", "ham"})
        raise ValueError(f"Labels must be 'spam' or 'ham'; unsupported labels: {', '.join(invalid)}.")

    result = pd.DataFrame({"text": text.map(normalize_text), "label": labels})
    result = result[result["text"].str.len() > 0].reset_index(drop=True)
    if result["label"].nunique() != 2 or len(result) < 8 or result["label"].value_counts().min() < 2:
        raise ValueError(
            "Dataset must contain at least eight non-empty emails and at least two examples of each label."
        )
    return result
