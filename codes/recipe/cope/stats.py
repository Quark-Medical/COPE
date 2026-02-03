import os
import json
import re
import argparse
from collections import defaultdict
import statistics

def get_file_idx(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1))

def process_file_group(file_list, base_path, domain_pattern):
    group_domain_rewards = defaultdict(list)
    group_all_rewards = []
    
    for file_name in file_list:
        file_path = os.path.join(base_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                data = json.loads(line)
                reward_p = data.get("real_personal_reward")
                reward_c = data.get("completeness_reward")
                reward = reward_p * reward_c
                if reward is None: 
                    raise ValueError("No 'real_personal_reward' field found in the JSON data.")
                
                monitor_info = data.get("monitor_info", {})
                eval_prompt = monitor_info.get("user_eval_prompt", "")
                if not eval_prompt: 
                    raise ValueError("No 'user_eval_prompt' field found in the JSON data.")
                    
                domains = domain_pattern.findall(eval_prompt)
                if not domains: 
                    raise ValueError("No 'Domain' found in the 'user_eval_prompt'.")
                
                unique_domains = list(set(d.strip() for d in domains))
                if len(unique_domains) > 1: 
                    raise ValueError("Multiple 'Domain' found in the 'user_eval_prompt'.")
                
                target_domain = unique_domains[0]
                val = float(reward)
                group_domain_rewards[target_domain].append(val)
                group_all_rewards.append(val)
    return group_domain_rewards, group_all_rewards

def analyze_data(base_path):
    domain_pattern = re.compile(r'\[Domain:\s*([^\]]+)\]')

    if not os.path.exists(base_path):
        print(f"Error: The path '{base_path}' does not exist.")
        return

    all_files = sorted([f for f in os.listdir(base_path) if f.endswith('.jsonl') and get_file_idx(f) >= 51 and get_file_idx(f) <= 101], key=get_file_idx)
    
    assert len(all_files) == 51, f"Expecting 51 files, actually found {len(all_files)}"
    
    groups = [all_files[i:i+17] for i in range(0, 51, 17)]
    
    group_summaries = []
    group_counts = []
    group_total_means = []

    for i, group_files in enumerate(groups):
        domain_rewards, all_rewards = process_file_group(group_files, base_path, domain_pattern)
        
        current_means = {dom: statistics.mean(res) for dom, res in domain_rewards.items()}
        current_counts = {dom: len(res) for dom, res in domain_rewards.items()}
        
        group_summaries.append(current_means)
        group_counts.append(current_counts)
        group_total_means.append(statistics.mean(all_rewards) if all_rewards else 0.0)
        
        print(f"Group {i+1} Processing complete. File range: {group_files[0]} ~ {group_files[-1]}")

    all_domains = sorted(group_summaries[0].keys())
    for i in range(1, 3):
        assert sorted(group_counts[i].keys()) == all_domains, f"The Domain type of Group {i+1} is inconsistent with the first group"

    print("\n" + "="*95)
    print(f"{'Domain':<20} | {'Count':<6} | {'G1 Mean':<10} | {'G2 Mean':<10} | {'G3 Mean':<10} | {'Cross-G Mean':<12} | {'Cross-G Std'}")
    print("-" * 95)

    cross_group_total_means = []
    
    for dom in all_domains:
        m1 = group_summaries[0][dom]
        m2 = group_summaries[1][dom]
        m3 = group_summaries[2][dom]
        
        three_means = [m1, m2, m3]
        avg_of_means = statistics.mean(three_means)
        std_of_means = statistics.stdev(three_means)
        
        count = group_counts[0][dom]
        
        print(f"{dom:<20} | {count:<6} | {m1:<10.4f} | {m2:<10.4f} | {m3:<10.4f} | {avg_of_means:<12.4f} | {std_of_means:.4f}")

    print("-" * 95)
    tm1, tm2, tm3 = group_total_means
    total_avg = statistics.mean(group_total_means)
    total_std = statistics.stdev(group_total_means)
    
    total_count_per_group = sum(group_counts[0].values())
    
    print(f"{'ALL DATA (Total)':<20} | {total_count_per_group:<6} | {tm1:<10.4f} | {tm2:<10.4f} | {tm3:<10.4f} | {total_avg:<12.4f} | {total_std:.4f}")
    print("="*95 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()
    analyze_data(args.path)
