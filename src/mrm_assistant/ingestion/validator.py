"""Quality checks for extracted PDF pages before indexing."""

from mrm_assistant.models import PageDocument, ValidationIssue, ValidationReport

MINIMUM_USEFUL_CHARACTERS = 40


def validate_pages(pages: list[PageDocument]) -> ValidationReport:
    """Flag pages that should not silently enter a retrieval index."""
    issues: list[ValidationIssue] = []
    for page in pages:
        if not page.text:
            issues.append(
                ValidationIssue(
                    document_name=page.document_name,
                    page_number=page.page_number,
                    severity="error",
                    message="No extractable text was found; OCR or manual review is required.",
                )
            )
        elif len(page.text) < MINIMUM_USEFUL_CHARACTERS:
            issues.append(
                ValidationIssue(
                    document_name=page.document_name,
                    page_number=page.page_number,
                    severity="warning",
                    message=(
                        "Very little text was extracted; verify that this page is not "
                        "scanned, visual-only, or incorrectly parsed."
                    ),
                )
            )

    return ValidationReport(
        total_pages=len(pages),
        error_count=sum(issue.severity == "error" for issue in issues),
        warning_count=sum(issue.severity == "warning" for issue in issues),
        issues=issues,
    )
