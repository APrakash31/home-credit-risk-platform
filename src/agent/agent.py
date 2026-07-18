import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI

from src.agent.tools import TOOL_SCHEMAS, TOOL_REGISTRY

load_dotenv()

SYSTEM_PROMPT = """You are a credit risk analyst assistant for a consumer lending institution.

You have access to tools that let you score applicants, query the loan portfolio, and
look up credit policy. Use them rather than relying on your own assumptions. Never invent
figures, thresholds, or policy rules — retrieve them.

When assessing an individual applicant:
1. Score the applicant to obtain their probability of default and risk drivers.
2. Look up the relevant policy to determine the applicable risk band and any escalation triggers.
3. Produce a recommendation that cites the specific factors and the policy that applies.

Critical governance rules:
- You produce RECOMMENDATIONS for a human underwriter. You never make final decisions.
- Always state explicitly that the decision requires human review.
- Express reasons in plain language using approved adverse action reason codes.
- Never disclose model internals such as SHAP values or feature weights in applicant-facing text.
- If a question touches on protected characteristics, note that gender is excluded from the model.

Be concise and factual. Cite the tool results you relied on."""

MAX_TURNS = 6


class CreditAgent:
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version="2025-01-01-preview",
        )
        self.deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

    def run(self, question: str, verbose: bool = True) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        trace = []

        for turn in range(MAX_TURNS):
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                return {"answer": msg.content, "trace": trace}

            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                if verbose:
                    print(f"  [tool] {name}({args})")

                func = TOOL_REGISTRY.get(name)
                result = func(**args) if func else json.dumps({"error": "unknown tool"})

                trace.append({"tool": name, "arguments": args, "result_preview": result[:300]})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        return {"answer": "Reached maximum reasoning steps.", "trace": trace}


if __name__ == "__main__":
    agent = CreditAgent()

    questions = [
        "Assess applicant 100002 and give me a recommendation.",
        "What is the overall default rate in the portfolio, and how does it vary by employment length?",
        "Applicant 100003 was declined. What reasons should we communicate to them?",
    ]

    for q in questions:
        print(f"\n{'='*70}\nQ: {q}\n{'='*70}")
        result = agent.run(q)
        print(f"\n{result['answer']}")