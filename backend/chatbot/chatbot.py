from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from .responses import build_risk_response, generic_advice


class Chatbot:
	"""
	Simple, production-friendly chatbot for teachers:
	- loads intents from intents.json
	- keyword-based matching with regex
	- can incorporate risk_tier signals (green/yellow/red)
	"""

	def __init__(self, intents_path: Path) -> None:
		self.intents = self._load_intents(intents_path)
		self.patterns = self._compile_patterns(self.intents)

	def _load_intents(self, path: Path) -> Dict:
		with open(path, "r") as f:
			return json.load(f)

	def _compile_patterns(self, intents: Dict) -> Dict[str, List[re.Pattern]]:
		compiled: Dict[str, List[re.Pattern]] = {}
		for intent in intents.get("intents", []):
			tag = intent["tag"]
			compiled[tag] = [re.compile(rf"\b{re.escape(p)}\b", re.IGNORECASE) for p in intent.get("patterns", [])]
		return compiled

	def _match_intent(self, text: str) -> Optional[str]:
		for tag, patterns in self.patterns.items():
			for pat in patterns:
				if pat.search(text):
					return tag
		return None

	def respond(self, text: str, risk_tier: Optional[str] = None) -> str:
		text = (text or "").strip()
		if not text:
			return "Hello! How can I help you support your students today?"

		# Risk-aware override
		if risk_tier:
			return build_risk_response(risk_tier)

		tag = self._match_intent(text)
		if tag is None:
			# Heuristics
			if any(k in text.lower() for k in ["risk", "dropout", "intervention"]):
				return build_risk_response("medium")
			return generic_advice()

		# Return the first canned response for the tag
		for intent in self.intents.get("intents", []):
			if intent["tag"] == tag and intent.get("responses"):
				return intent["responses"][0]
		return generic_advice()
