from dataclasses import dataclass


@dataclass
class ReportingConfig:
    generate_html_report: bool = True
    report_title: str = "My XAI Experiment Report"
