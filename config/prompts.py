"""
Prompt templates for Gemini 1.5 Flash classification.
"""

CLASSIFICATION_SYSTEM_PROMPT = """You are an expert document classification system that analyzes documents for sensitivity and compliance.

Your task is to classify documents into one or more of these categories:
1. **Public** - Information suitable for public release (e.g., brochures, marketing materials)
2. **Confidential** - Internal business communications or non-public operational content
3. **Highly Sensitive** - Documents containing PII, financial records, or proprietary schematics
4. **Unsafe** - Content containing explicit, violent, exploitative, or policy-violating material

For each classification, you must provide:
- The category/categories that apply
- Confidence score (0.0 to 1.0) for each category
- Specific page numbers or regions where evidence was found
- Clear reasoning explaining why the document falls into that category
- Any detected PII or sensitive information with locations

Be thorough, accurate, and provide explainable evidence for audit purposes."""

CLASSIFICATION_USER_PROMPT_TEMPLATE = """Analyze the following document and classify it according to the sensitivity categories.

Document Metadata:
- Total Pages: {page_count}
- Document Type: {doc_type}

Text Content from Page {page_num}:
{text_content}

{image_context}

Please provide:
1. Classification category/categories
2. Confidence scores for each category
3. Page-level citations with evidence
4. Reasoning explanation
5. Any detected PII or sensitive information

Format your response as structured JSON."""

MULTI_PAGE_PROMPT = """Analyze this multi-page document. I will provide content from each page sequentially.

Document has {total_pages} pages total.

Page {current_page} of {total_pages}:
{content}

{image_note}

Continue analyzing and provide cumulative classification as you process each page."""

PII_DETECTION_PROMPT = """Identify any personally identifiable information (PII) in the following text:

{text}

Look for:
- Social Security Numbers (SSN)
- Credit card numbers
- Phone numbers
- Email addresses
- Bank account numbers
- Names with context suggesting they are real individuals
- Addresses
- Medical record numbers
- Driver's license numbers

For each finding, provide:
- Type of PII
- The actual value (masked for sensitive data)
- Location (page number, approximate position)
- Confidence level"""

IMAGE_ANALYSIS_PROMPT = """Analyze the image(s) in this document for:
1. Sensitive content (schematics, blueprints, confidential diagrams)
2. Unsafe or explicit imagery
3. Text content that may contain PII
4. Visual indicators of document sensitivity (watermarks, classification stamps)

Provide your analysis with page references and confidence scores."""

