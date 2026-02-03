import json
import aiohttp
import asyncio
from openai import AsyncOpenAI

def render_prompt(assistant_response: str):
    render_template = """<|im_start|>user
{}<|im_end|>
<|im_start|>assistant<think>

</think>

"""
    return render_template.format(assistant_response)

async def request_vllm_async(content_str, apikey, model="Qwen3-8B", env='test',
                       session_id=None, temperature=0.8, max_token=1024, do_sample=True,
                       top_p=1.0, top_k=-1, max_try=5, retry_interval=1):

    url = 'http://xxxxxxxxxxxxxxxxxxxxxxxxxxx'

    authorization = 'Bearer ' + apikey
    headers = {
        'Content-Type': 'application/json',
        'Authorization': authorization
    }

    body = {
        "session_id": "hf-xxxxxxxxxxxxxxxxxxxxxxxxx" if session_id is None else session_id,
        "request_id": "hf-xxxxxxxxxxxxxxxxxxxxxxxxx",
        "model": model,
        "prompt": content_str,
        "source": {
            "ori_query": "1234"
        },
        'extra_args': {
            'max_new_tokens': max_token,
            'top_p': top_p,
            'top_k': top_k,
            'logprobs': True,
            'temperature': temperature,
            'top_p_decay': 0.0,
            'top_p_bound': 0.0,
            'add_BOS': False,
            'stop_on_double_eol': False,
            'stop_on_eol': False,
            'prevent_newline_after_colon': False,
            'random_seed': 42,
            'no_log': True,
            'stop_token': 50256,
            'length_penalty': 1.0,
            'do_sample': do_sample,
            'no_repeat_ngram_size': 0,
            'beam_width': 1
        }
    }

    async with aiohttp.ClientSession() as session:
        for attempt in range(1, max_try + 1):
            try:
                async with session.post(
                    url,
                    data=json.dumps(body, ensure_ascii=False).encode('utf-8'),
                    headers=headers
                ) as resp:
                    if resp.status != 200:
                        print(f"[Try {attempt}/{max_try}] Request failed: HTTP {resp.status}")
                    else:
                        data = await resp.json(content_type=None)
                        res = data["choices"][0]["message"]["content"]
                        
                        if res and res.strip():
                            return res
                        else:
                            print(f"[Try {attempt}/{max_try}], return content is empty (space or empty string)")
            except Exception as e:
                print(f"[Try {attempt}/{max_try}] request vllm error: {e}")

            if attempt < max_try:
                await asyncio.sleep(retry_interval)

    return None


async def request_bailian_async(content_str, apikey, model, enable_thinking=False, max_try=5, retry_interval=1):
    client = AsyncOpenAI(
    api_key=apikey,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    for attempt in range(max_try):
        try:
            completion = await client.chat.completions.create(
                extra_body={"enable_thinking": enable_thinking},
                stream=False,
                model=model,
                messages=[
                    {'role': 'user', 'content': content_str}
                ],
                timeout=60.0
            )
            
            content = completion.choices[0].message.content
            
            if isinstance(content, str) and content.strip():
                return content
            
            print(f"Attempt {attempt + 1}: Content is empty or format is incorrect, preparing to retry...")

        except Exception as e:
            print(f"The {attempt + 1}th attempt occurred an exception: {e}")

        if attempt < max_try:
            await asyncio.sleep(retry_interval)
    
    print("Reached maximum retry count, failed to call the Bailian platform.")
    return None

async def async_request(content_str, apikey, mode, **kwargs):
    if mode == 'vllm':
        rendered_content_str = render_prompt(content_str)
        vllm_params = {
            'content_str': rendered_content_str,
            'apikey': apikey,
            'model': kwargs.get('model', "model-name-here"),
            'env': kwargs.get('env', 'test'),
            'session_id': kwargs.get('session_id'),
            'temperature': kwargs.get('temperature', 0.8),
            'max_token': kwargs.get('max_token', 1024),
            'do_sample': kwargs.get('do_sample', True),
            'top_p': kwargs.get('top_p', 1.0),
            'top_k': kwargs.get('top_k', -1),
            'max_try': kwargs.get('max_try', 5),
            'retry_interval': kwargs.get('retry_interval', 1)
        }
        return await request_vllm_async(**vllm_params)
    elif mode == 'bailian':
        bailian_params = {
            'content_str': content_str,
            'apikey': apikey,
            'model': kwargs.get('model', "qwen-flash"),
            'enable_thinking': kwargs.get('enable_thinking', False),
            'max_try': kwargs.get('max_try', 5),
            'retry_interval': kwargs.get('retry_interval', 1)
        }
        return await request_bailian_async(**bailian_params)
    else:
        raise ValueError("The mode must be 'vllm' or 'bailian'")