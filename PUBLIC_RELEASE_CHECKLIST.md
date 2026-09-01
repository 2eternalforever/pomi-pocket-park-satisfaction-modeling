# Public release checklist

Before creating a public release, verify that:

- no names, email addresses, affiliations, account identifiers, or other personal identifiers are present;
- no direct respondent identifiers or re-identifiable free-text fields are included;
- participant-level data have been anonymized and approved for the intended access level;
- variable scales and recoding rules are documented;
- temporary files, Python bytecode, and local environments are excluded;
- every individual file is below the hosting service's size limit;
- the workflow runs from a clean environment using `requirements.txt`;
- generated CSV values match the reference outputs;
- the release metadata accurately describes the deposited files.

The participant-level dataset is intentionally absent from this package. Add it only after completing the privacy, consent, and ethics checks above.
