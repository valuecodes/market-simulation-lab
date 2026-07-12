# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for security vulnerabilities.

Instead, report privately via GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability):
open the repository's **Security** tab and choose **Report a vulnerability**.
(Enable "Private vulnerability reporting" in the repository's Security settings
if it is not already on.)

Please include:

- a description of the issue and its impact,
- steps to reproduce, and
- any suggested remediation.

We aim to acknowledge reports within a few days.

## Scope

This is a **local-first research tool** with no accounts, database, or stored
credentials. When run locally against your own data, the attack surface is
minimal. The area most worth scrutiny is **untrusted CSV upload handling**
(`src/portfolio_research_lab/data.py`), which is bounded and validated before
any computation. If you intend to expose the Streamlit app to the public
internet, review `README.md` for the additional hardening that entails.
