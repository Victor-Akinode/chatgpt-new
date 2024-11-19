from __future__ import annotations
import os
import pandas as pd
import openai
import asyncio
import logging
import aiolimiter
from tqdm.asyncio import tqdm_asyncio
from aiohttp import ClientSession
from typing import Any, List

def create_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Function to create prompts in the original style
def create_prompt(question, choices, prompt_style=1):
    options_text = "\n".join([f"{i}) {choice}" for i, choice in enumerate(choices)])
    if prompt_style == 1:
        return f"""Answer the following question by selecting the correct option:

Question: {question}

{options_text}

Your answer:"""
    elif prompt_style == 2:
        return f"""Please choose the correct answer for the following question:

{question}

Options:
{options_text}

Answer:"""
    elif prompt_style == 3:
        return f"""Let's test your knowledge! Pick the correct answer:

{question}

{options_text}

Your choice:"""
    elif prompt_style == 4:
        return f"""In this examination, answer the following question by selecting the correct option from the numbered choices:

{question}

Options:
{options_text}

Choose your answer:"""
    else:
        return f"""Read the following question and select the best answer from the options provided:

{question}

{options_text}

Your response:"""

async def _throttled_openai_chat_completion_acreate(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    async with limiter:
        for _ in range(100):
            try:
                return await openai.ChatCompletion.acreate(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.error.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 20 seconds."
                )
                await asyncio.sleep(40)
            except asyncio.exceptions.TimeoutError:
                logging.warning("OpenAI API timeout. Sleeping for 20 seconds.")
                await asyncio.sleep(40)
            except openai.error.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                break
        return {"choices": [{"message": {"content": ""}}]}

async def generate_from_openai_chat_completion(
    full_contexts: List[List[dict[str, str]]],
    model_config: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    requests_per_minute: int = 300,
) -> list[str]:
    session = ClientSession()
    openai.aiosession.set(session)
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=model_config,
            messages=full_context,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for full_context in full_contexts
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    await session.close()
    return [x["choices"][0]["message"]["content"] for x in responses]

def extract_answer(response):
    # Extracting the answer from the AI response
    for option in ['0', '1', '2', '3']:
        if option in response:
            return int(option)
    return None

def evaluate_predictions(predictions, correct_answers):
    correct = 0
    for pred, correct_answer in zip(predictions, correct_answers):
        if pred == correct_answer:
            correct += 1
    return correct / len(correct_answers) * 100

if __name__ == '__main__':
    openai.api_key = os.getenv("OPENAI_API_KEY")
    model_name = 'gpt-4'
    output_dir = 'results/'

    create_dir(output_dir)
    
    # Load the new dataset
    df = pd.read_csv('data/questions.csv')
    
    all_input_messages = []
    correct_answers = []
    
    for i in range(df.shape[0]):
        question = df['question'].iloc[i]
        choices = eval(df['choices'].iloc[i])  # Convert string list to actual list
        answer = df['answer'].iloc[i]
        
        # Cycle through the prompt styles or choose randomly
        prompt = create_prompt(question, choices, prompt_style=(i % 5) + 1)
        input_message = [{"role": "user", "content": prompt}]
        all_input_messages.append(input_message)
        correct_answers.append(answer)
    
    # Generate responses from the AI model
    responses = asyncio.run(generate_from_openai_chat_completion(
        all_input_messages, 
        model_name, 
        temperature=0.3, 
        max_tokens=100, 
        top_p=1.0, 
        requests_per_minute=300
    ))
    
    # Extract predictions from responses
    predictions = [extract_answer(response) for response in responses]
    
    # Evaluate accuracy
    accuracy = evaluate_predictions(predictions, correct_answers)
    
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Save results
    df['gpt-4-predictions'] = predictions
    df.to_csv(output_dir + 'results.csv', index=False)