from openai import OpenAI
import tomllib
import csv
from rich import print
from pathlib import Path
import pandas as pd

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

client = OpenAI(api_key=config['api']['api_key'], base_url=config['api']['base_url'])

output_path = Path(config['output']['file_name'])
already_tested = []

if output_path.exists() and output_path.is_file():
    df = pd.read_csv(output_path)
    already_tested = df['Model'].to_list()
else:
    df = pd.DataFrame(columns=['Model', 'Response', 'Total Tokens', 'Prompt Tokens', 'Completion Tokens'])

for model in config['testing']['models']:
    if model not in already_tested:
        try: 
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": config['testing']['prompt']}
                ],
            )

            response_str = str(completion.choices[0].message.content)

            if completion.usage is not None:
                print(f"[cyan]{model}[reset]: {response_str} [purple]({completion.usage.total_tokens} Tokens)")
                data_list = [model, response_str, completion.usage.total_tokens, completion.usage.prompt_tokens, completion.usage.completion_tokens]
            else:
                print(f"[cyan]{model}[reset]: {response_str}")
                data_list = [model, response_str, 'N/A', 'N/A', 'N/A']

        except Exception as e:
            data_list = [model, 'N/A', 'N/A', 'N/A', 'N/A']
            print(f'[red]Error![reset] "{e}" when processing with model [cyan]{model}')

        df = pd.concat([pd.DataFrame([data_list], columns=df.columns), df], ignore_index=True)

print(df)
df.to_csv(output_path, index=False)
