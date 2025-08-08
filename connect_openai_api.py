from openai import OpenAI

client = OpenAI(
  api_key="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"
)

response = client.responses.create(
  model="gpt-4o-mini",
  input="Assess whether these messages are red flags or neutral/green flags.",
  store=True,
)

print(response.output_text)
