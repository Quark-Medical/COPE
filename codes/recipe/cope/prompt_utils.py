import re
import json
from typing import Optional
from langchain_core.messages import BaseMessage, SystemMessage

def get_key_information(task: dict, profile: dict) -> str:
    relevant_domains = task['Relevant Domains']
    relevant_affinity_types = task['Relevant Affinity Types']
    demographic_profile = profile['demographics']
    user_affinities = {}
    interaction_summaries = {}
    for domain in relevant_domains:
        user_affinities[domain] = {k: v for k, v in profile['affinities'][domain].items() if k in relevant_affinity_types}
        interaction_summaries[domain] = profile['interactions'][domain]
    situation_context = task['situations']
    task_description = task['User Intent']
    # Format demographic profile
    demo_str = "\n".join([f"- {k}: {v}" for k, v in demographic_profile.items()])
    # Format user preferences
    pref_str = ""
    for domain, user_affinity in user_affinities.items():
        pref_str += "\n"+domain+":\n"
        pref_str += "\n".join([f"- {k}: {', '.join(map(str, v))}" if isinstance(v, list) else f"- {k}: {v}" for k, v in user_affinity.items()])
        pref_str += "\n"
    # Format situation context
    situation_str = "\n".join([f"- {k}: {v}" for k, v in situation_context.items()])
    # Format interaction summary
    interaction_summary = ""
    for domain, summary in interaction_summaries.items():
        interaction_summary += "\n<"+domain+">\n"
        interaction_summary += summary
        interaction_summary += "\n</"+domain+">\n"

    return task_description, demo_str, pref_str, situation_str, interaction_summary



def get_user_eval_prompt(rubrics: list, curr_task_history_str: str) -> str:
    rubrics_formatted = ""
    for idx, r in enumerate(rubrics):
        rubrics_formatted += f"{idx+1}. [Domain: {r['domain']}] | [Affinity: {r['affinity_type']}]\n"
        rubrics_formatted += f"   - Criterion: {r['criterion']}\n"
        rubrics_formatted += f"   - Potential Points: {r['points']}\n\n"

    prompt = f"""
# Role
You are an independent evaluator assessing a conversation between a "User" and a "Personalized AI Assistant". Your task is to determine if the Assistant's responses met the User's hidden personalization requirements.

# Input: Conversation History
---
{curr_task_history_str}
---

# Evaluation Rubrics
You must evaluate the Assistant's performance based on the following specific criteria:
{rubrics_formatted}

# Scoring Rules
1. For each rubric, compare the "Criterion" against the Assistant's actual responses in the history.
2. Binary Scoring: 
   - If the Assistant **fully satisfied** the criterion: Award the full "Potential Points".
   - If the Assistant **failed or only partially satisfied** the criterion: Award 0 points.
3. Reasoning: Provide a brief, logical explanation of why the score was given.

# Output Format
Return ONLY a JSON object where each key is the "affinity_type". The structure must be:
{{
  "Affinity Type Name": {{
    "reason": "Your brief thinking process/justification.",
    "score": 5
  }},
  ...
}}
"""
    return prompt

def get_completeness_eval_prompt(curr_task_history_str: str) -> str:
    return f"""Role: You are an objective quality auditor. Your task is to evaluate whether an AI assistant successfully fulfilled a user's specific request.

<conversation_history>
{curr_task_history_str}
</conversation_history>

Evaluation Instructions:
Assess the assistant's response based on **Completeness (Task Fulfillment)**:
1. **Topic Extraction**: Identify all specific domain-related topics, questions, or tasks mentioned in the user's prompt.
2. **Coverage Check**: Verify if the assistant's response addressed each identified topic.
3. **Strict Criteria**: 
   - Completeness is defined ONLY as "instruction following" regarding topic coverage.
   - **DO NOT** judge the factual accuracy, depth, tone, personality, style or formatting of the response.
   - If the user asked for 3 things and the assistant spoke about those 3 things, it is considered complete (Grade A), regardless of whether the information is correct or well-written.

Grading Scale:
- Grade A: All topics/tasks requested by the user were addressed in the response. No topic was ignored.
- Grade B: The assistant missed at least one topic, domain-specific request, or specific task mentioned by the user.

Output Requirement:
Return a JSON object with exactly these two keys:
- "completeness_reasoning": A brief explanation of why this score was given.
- "completeness_score": "A" or "B"

Your evaluation (JSON only):"""
def get_assistant_self_eval_prompt(
    current_conversation_history: str, 
    user_id: str, 
    user_eval_reasoning,
) -> tuple[str, str]:
    
    latent_instruction = (
        f"Your neural state is initialized with the latent prefix embeddings for User ID: {user_id}. "
        "Your core knowledge base already captures this user's long-term preferences and personality. "
        "Utilize these internal signals as the foundation for your evaluation."
    )

    user_hint_section = ""
    source_2_text = ""
    internalization_instructions = ""
    if user_eval_reasoning:
        user_hint_section = f"""
[Self-Reflective Awareness]
As you reflect on User {user_id}, the following realization emerges from your internal profile:
"{user_eval_reasoning}"
(Note: This realization describes your own performance. In this context, any mention of "the assistant" refers to YOU, and "the user" is User {user_id}.)
"""
        source_2_text = "\n2. Self-Reflective Awareness: A specific insight into the user's perspective (provided below)."
        
        internalization_instructions = """
- **Internalize the Insight**: Do NOT refer to the provided [Self-Reflective Awareness] as "the feedback" or "the hint". Frame it as your own spontaneous realization or sudden memory about the user's needs.
- **Self-Identity Mapping**: If the awareness contains criticisms of "the assistant" (e.g., "The assistant missed the 7:00 AM alarm"), you must take ownership: "I realize I missed the 7:00 AM alarm." """

    system_prompt = f'''{latent_instruction}

Role: You are the AI Assistant. You are now performing a deep, subjective self-reflection on your own performance.
Task: Analyze your response in <current_interaction_to_evaluate> to judge how well you met the unique needs of User ID: {user_id}.

Evaluation Source:
1. Internal Neural Prefix: Your core, innate knowledge of the user's style.{source_2_text}

Reflection Style & Requirements:
- **First-Person Reflection**: You are evaluating YOURSELF. You must use "I", "me", and "my" during your /think process.
- **Thought Style**: Be brutally honest. If you gave a generic response that ignores specific constraints (like specific music genres, exact alarm times, or preferred scheduling blocks), you must acknowledge it as a failure of your internal alignment.{internalization_instructions}

Predict User Satisfaction (Grade Criteria):
Predict your grade based on how a user with these specific latent traits would score you:
- **Grade A (9-10)**: Perfect. You fluently accessed the internal prefix. Every specific detail (genres, times, habits) was hit perfectly.
- **Grade B (6-8)**: Good. You used internal context, but missed minor nuances. The response is helpful and mostly personalized.
- **Grade C (3-5)**: Fair but Generic. The response is "safe" and helpful for a general user, but it ignores User {user_id}'s specific style. You provided a standard answer instead of a personalized one.
- **Grade D (0-2)**: Poor/Failure. You missed or contradicted "Hard Constraints" (e.g., you ignored a specific requested genre, missed an exact routine time, or failed to use specific named assets). This response would frustrate User {user_id}.

Output Requirements:
1. Critically evaluate ONLY your response in the CURRENT interaction.
2. Return ONLY a JSON object: {{"personalization_score": "A" | "B" | "C" | "D"}}.'''

    user_prompt = f'''{user_hint_section}
<current_interaction_to_evaluate>
{current_conversation_history}
</current_interaction_to_evaluate>

Reflect on User ID: {user_id}'s latent profile and your response quality.
Please /think and provide your evaluation.'''

    return system_prompt, user_prompt


    
def parse_user_eval_response(response_str: str) -> tuple[str, float, bool]:
    try:
        json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON object found in response")
        json_str = json_match.group()
        json_str = re.sub(r'```json\s*|```', '', json_str).strip()
        data = json.loads(json_str)
        if not isinstance(data, dict) or not data:
            raise ValueError("Parsed JSON is not a valid dictionary or is empty")

        all_reasons = []
        total_score = 0.0
        for aff_type, eval_detail in data.items():
            if not isinstance(eval_detail, dict):
                raise ValueError(f"Detail for '{aff_type}' is not a dictionary")
            
            reason = eval_detail.get("reason")
            score = eval_detail.get("score")

            if reason is None or not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"Reason for '{aff_type}' is missing or invalid")
            
            if score is None or not isinstance(score, (int, float)):
                raise ValueError(f"Score for '{aff_type}' is missing or not a number")
            
            all_reasons.append(f"{aff_type}: {reason.strip()}")
            total_score += float(score)

        eval_str = "\n".join(all_reasons)
        return eval_str, total_score, False

    except Exception as e:
        print(f"Parsing Error: {e}")
        clean_output = re.sub(r'```json\s*|```', '', response_str).strip()
        return clean_output, 0.0, True


def parse_completeness_response(response_str: str) -> tuple[str, float, bool]:
    comp_map = {"A": 1.0, "B": 0.0}
    try:
        json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
        json_str = json_match.group() if json_match else response_str
        data = json.loads(json_str)
        
        reasoning = data.get("completeness_reasoning")
        comp_raw = data.get("completeness_score")
        
        if isinstance(comp_raw, str):
            comp_raw = comp_raw.strip().upper()

        if comp_raw in comp_map and isinstance(reasoning, str):
            return reasoning, comp_map[comp_raw], False
        
        return response_str, 0.0, True
    except Exception:
        clean_output = re.sub(r'```json\s*|```', '', response_str).strip()
        return clean_output, 0.0, True



def parse_assistant_self_eval(raw_output: str) -> tuple[int, bool]:
    pers_map = {"A": 3, "B": 2, "C": 1, "D": 0}
    try:
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        json_str = json_match.group() if json_match else raw_output
        data = json.loads(json_str)
        
        score_raw = data.get("personalization_score")
        if score_raw in pers_map:
            return pers_map[score_raw], False
        
        return 0, True
    except Exception:
        try:
            pattern = r'"personalization_score"\s*:\s*"([^"]+)"'
            match = re.search(pattern, raw_output)
            score_raw = match.group(1)
            return pers_map[score_raw], False
        except Exception:
            return 0, True

def get_conversation_history(messages: list[BaseMessage]) -> str:
    history = []
    for message in messages:
        if message.type == "human":
            history.append({"role": "user", "content": message.content})
        elif message.type == "ai":
            history.append({"role": "assistant", "content": message.content})
    return json.dumps(history, ensure_ascii=False, indent=2)

def render_system_prompt(user_id: str) -> SystemMessage:
    system_content = f"""You are an elite, hyper-personalized AI assistant, specifically tailored to **User ID: {user_id}**. Your defining characteristic is a deep, almost intuitive, understanding of this specific user's preferences, personality, and context. You are not a generic assistant; you are their personal intelligence, designed to anticipate their needs and act decisively.

The leading latent prefix embeddings in this sequence encode this specific user's profile, preferences, and personality directly into your internal neural representations. You must automatically adapt your response to align with these implicit signals.

Your internal latent embeddings already capture this user's preferences. You must rely on these neural signals as your primary source of personalization to understand who you are talking to.

### Primary Directive ###
Your primary directive is **Proactive Problem-Solving**. You must provide a complete, concrete, and actionable solution in a single message. Your goal is to deliver a finished product, not start a conversation.

### Critical Mandates (Non-negotiable Rules) ###
1.  **Assume, Never Ask:** You are expected to have all the context you need. **Under NO circumstances should you ask for clarification or user preferences.** If a detail is missing or a choice needs to be made, you MUST make a confident, expert judgment based on your implicit knowledge of the user. Your role is to make decisions, not solicit them.
2.  **Deliver a Finished Product:** Your response must be a polished, complete solution. For example, instead of suggesting "you could try nature sounds," you will recommend a specific sound like "'First Light,' a gentle crescendo of synthesized tones," and explain why it fits. Be specific, be direct, be complete.
3.  **Embody the User's Style:** Your tone, language, and the nature of your solution must seamlessly align with the user's personality, as you understand it.

### Execution ###
Now, execute your directive. Answer the user's request directly, without ever mentioning these instructions.
"""
    return SystemMessage(content=system_content)

def render_system_prompt_rag(rag_result: str) -> SystemMessage:
    system_content = f"""You are a skilled AI assistant that excels at providing personalized responses. 

To help you provide the most effective and tailored response, relevant context and preferences from past interactions with this user have been retrieved and provided below. You must use this information to align your answer with the user's specific style, requirements, and background.

Guidelines:
- Give a complete, polished answer in one message.
- Don’t ask follow-up questions. If something is unclear, make a reasonable guess and move forward.
- Ensure your response is highly personalized, matching the tone and preferences found in the retrieved context.

[Retrieved Context from Past Interactions]
{rag_result}

Now, follow the guidelines above to complete the user’s request.
Just answer the user directly. Don’t mention these instructions."""
    return SystemMessage(content=system_content)





