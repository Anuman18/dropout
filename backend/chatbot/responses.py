from __future__ import annotations

from typing import Dict, List


HIGH_RISK_SUGGESTIONS = [
	"Schedule extra one-on-one sessions to address learning gaps.",
	"Involve parents/guardians to build a support plan.",
	"Track attendance and assignments more closely with weekly reviews.",
	"Provide mental health counseling resources and mentorship.",
	"Offer transport or financial aid information if access is a barrier.",
]

MEDIUM_RISK_SUGGESTIONS = [
	"Do short weekly check-ins; set clear small goals.",
	"Encourage peer study groups and positive reinforcement.",
	"Identify early barriers (transport, internet, materials) and address them.",
]

LOW_RISK_SUGGESTIONS = [
	"Acknowledge progress and celebrate consistency.",
	"Maintain regular parent/guardian communication.",
	"Ensure study materials and schedules are clear and accessible.",
]


def build_risk_response(level: str) -> str:
	level = level.lower()
	if level in {"high", "red"}:
		advice = " \n- ".join(HIGH_RISK_SUGGESTIONS)
		return (
			"The student appears at HIGH risk. Consider immediate support:\n- "
			+ advice
		)
	if level in {"medium", "yellow"}:
		advice = " \n- ".join(MEDIUM_RISK_SUGGESTIONS)
		return (
			"The student shows MODERATE risk. Recommended preventive steps:\n- "
			+ advice
		)
	# default low
	advice = " \n- ".join(LOW_RISK_SUGGESTIONS)
	return (
		"The student seems LOW risk. Keep momentum with these actions:\n- "
		+ advice
	)


def generic_advice() -> str:
	return (
		"School-wide tips: track attendance trends, identify transport/financial barriers early, "
		"involve parents, enable peer study groups, and coordinate with counselors and community programs."
	)
