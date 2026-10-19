"""SOAR / SIEM integration sketches for Drishti-PAM.

This package ships reference artefacts that connect the detector
output to a SOC stack:

    splunk_detection.spl      Splunk SPL search over rule_predictions.json
    sentinel_rule.kql         Microsoft Sentinel KQL over the same shape
    xsoar_playbook.json       Cortex XSOAR v6 playbook stub (5-task spine)
    _validate.py              stdlib syntactic-check for the three above

The artefacts are reference sketches, not production rules. They are
syntactically valid (SPL pipe-balance, KQL extend/where/project
structure, XSOAR v6 minimum-required schema); test them in your own
Splunk, Sentinel, or XSOAR environment before production use.

The companion `Dockerfile` at the repository root packages the
static dashboard so a SOC engineer can `docker build` + `docker run`
and inspect the per-session triage view without installing Python.
"""

__all__ = ["_validate"]
