# 🧠 AI-Powered Regulatory Document Classifier (Gemini Version)

## 📄 Project Overview
This project builds an intelligent backend system that automatically classifies multi-page, multi-modal documents into:
- **Public**
- **Confidential**
- **Highly Sensitive**
- **Unsafe**

It uses **Gemini 1.5 Flash** for reasoning and evidence generation, with optional dual-model verification for self-consistency.  
The system outputs **explainable, citation-based results** for compliance and audit workflows.

---

## 🎯 Problem Description
Organizations handle thousands of documents daily, ranging from marketing materials to internal memos and defense schematics.  
Manual review for confidentiality or compliance is slow, inconsistent, and prone to human error.

The goal is to **automate document sensitivity classification** using a hybrid of deterministic policy rules and AI reasoning — producing human-auditable outputs with precise citations.

---

## 🧠 Core Challenge
- Extracting and understanding **text + images** from multi-page PDFs.  
- Detecting **personally identifiable information (PII)** such as SSNs or account numbers.  
- Recognizing **confidential content** like internal memos or proprietary designs.  
- Identifying **unsafe or restricted** imagery and phrasing.  
- Providing **transparent evidence and reasoning** for every decision.

---

## ⚙️ System Objectives
1. **Ingestion & Preprocessing:** Extract text/images and metadata from PDFs using OCR.  
2. **Classification & Reasoning:** Use Gemini 1.5 Flash with a configurable policy tree for sensitivity labeling.  
3. **Evidence Tracing:** Highlight and cite pages or regions supporting each classification.  
4. **Verification:** Perform dual-LLM self-checks for reduced human validation.  
5. **Reporting:** Export structured JSON and PDF summaries.  
6. **HITL Feedback:** Collect user corrections and retrain prompt logic.

---

## 🧩 Project Folder Structure

