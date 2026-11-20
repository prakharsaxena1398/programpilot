# Program Charter: LoanDesk AI Assistant

## 1. Background

SahajFinance, a consumer lending NBFC, serves 1.8 million retail borrowers
through a contact-center network and a borrower web portal. Borrowers
currently resolve loan-account questions by calling the support line or by
visiting a branch. Average handle time is 14 minutes per call, and agents
frequently misquote repayment schedules, generating complaints and
avoidable goodwill waivers. Leadership has approved a borrower-facing AI
assistant, branded LoanDesk, that answers loan-account and eligibility
questions in natural language, 24 hours a day.

## 2. Program Vision

LoanDesk will be a conversational chatbot embedded in the borrower portal.
It will answer questions such as "What is my outstanding after this month's
payment, and can I prepay next quarter?" by combining a retrieval-augmented
generation (RAG) pipeline over our loan product documents with our
account servicing database. To improve answer quality beyond generic
prompting, the team proposes to fine-tune a custom model on historical
support transcripts from the last two years. The assistant will be built as
an autonomous agent that can look up account balances, retrieve repayment
schedules, and place draft payment instructions on behalf of the borrower
without waiting for an operator. We will rely on OpenAI as our LLM vendor
under their standard subscription plan, since the portal is already a
single-tenant deployment and negotiating a second provider is out of scope
for this fiscal year.

## 3. Scope

Phase 1 covers the borrower portal chat widget, an account-data ingestion
pipeline that will integrate with our loan management system in real time to
keep balances and schedules current, and the support-transcript training
corpus. Borrowers will paste or forward their customer emails, which contain
account numbers, names, phone numbers, and payment data from deposit
confirmations, directly into the chat so the assistant can check loan status
during the conversation. The pilot covers 5,000 borrowers in the North
region. We plan to pilot in 6 weeks from kickoff, including data pipeline
build, prompt engineering, and borrower onboarding. Production rollout to
all regions follows immediately after a successful pilot.

## 4. Success Measures

Success will be measured through borrower satisfaction, CSAT, and adoption
of the assistant as the primary support channel. The program is considered
successful when borrowers tell us the answers feel trustworthy and the
support team reports a lighter call volume. Marketing will survey
sentiment after launch.

## 5. Team and Governance

The core team is two engineers borrowed from the portal team, one data
scientist at 50 percent allocation, and a product manager. The data
scientist is also committed to the collections-optimization program through
the same quarter. No dedicated budget line exists for model quality
checks; the team believes borrower feedback during the pilot will be
sufficient to judge answer quality. Complaints about wrong answers will be
handled by the same two engineers who built the system.

## 6. Risks Considered

The team has reviewed legal requirements at a high level and considers
data handling low risk because the borrower relationship is contractual.
No formal privacy review is planned before the pilot since the assistant
only reads borrower-supplied content. Vendor terms are accepted as-is.
