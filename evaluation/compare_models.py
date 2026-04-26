import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load from existing project
from evaluation.benchmark_models import run_benchmark_episode
from finsense.memory import MemorySystem

load_dotenv()

def plot_comparison(results, base_model_name, output_path="model_comparison_graphs.png"):
    base_scores = results[base_model_name]["scores"]
    base_rewards = results[base_model_name]["rewards"]
    
    trained_name = os.getenv("MODEL_NAME", "Trained Model")
    trained_scores = results[trained_name]["scores"]
    trained_rewards = results[trained_name]["rewards"]
    
    episodes = np.arange(1, len(base_scores) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Scores
    width = 0.35
    ax1.bar(episodes - width/2, base_scores, width, label='Base (Untrained)', color='#ef4444')
    ax1.bar(episodes + width/2, trained_scores, width, label='FinSense (Trained)', color='#22c55e')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('FinSense Score (0-1)')
    ax1.set_title('Task Score Comparison (Higher is Better)')
    ax1.set_xticks(episodes)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Total Rewards
    ax2.plot(episodes, base_rewards, marker='o', label='Base (Untrained)', color='#ef4444', linewidth=2)
    ax2.plot(episodes, trained_rewards, marker='s', label='FinSense (Trained)', color='#22c55e', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Cumulative Reward')
    ax2.set_title('RL Reward Comparison (Higher is Better)')
    ax2.set_xticks(episodes)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\n[DONE] Saved comparison graphs to: {output_path}")

def run_comparison(episodes: int, task_id: str):
    from hf_wrapper import HFWrapper as OpenAI
    
    hf_token = os.getenv("HF_TOKEN")
    trained_api_base = os.getenv("API_BASE_URL")
    trained_model_name = os.getenv("MODEL_NAME")
    
    if not hf_token or not trained_api_base:
        print("Error: Please ensure HF_TOKEN and API_BASE_URL are in your .env file.")
        return
    print("Using Dedicated Endpoints for BOTH models. This should be extremely fast and reliable!")
    
    import openai
    from hf_wrapper import HFWrapper
    
    # Base Model Client (Dedicated Endpoint running vLLM)
    base_client = openai.OpenAI(
        api_key=hf_token,
        base_url="https://d2ilzm11a6qbzoc1.us-east-1.aws.endpoints.huggingface.cloud/v1/",
        timeout=120.0,
        max_retries=2
    )
    base_model_name = "Qwen/Qwen2.5-7B-Instruct"

    # Trained Model Client (Dedicated Endpoint running HF Text Generation)
    trained_client = HFWrapper(
        api_key=hf_token,
        base_url=trained_api_base
    )
    
    print(f"============================================================")
    print(f" COMPARING UNTRAINED VS TRAINED MODEL (MEMORY=OFF)")
    print(f" Task: {task_id} | Episodes: {episodes}")
    print(f"============================================================\n")
    
    models_to_test = [
        ("Base (Untrained)", base_model_name, base_client),
        ("FinSense (Trained)", trained_model_name, trained_client)
    ]
    
    # ---------------------------------------------------------
    # TEST CONNECTIONS FIRST
    # ---------------------------------------------------------
    print("Testing connections and waking up models. This may take a minute...\n")
    for label, model_name, client in models_to_test:
        print(f"[*] Testing {label} ({model_name})...", end=" ", flush=True)
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Hello, are you awake? Please reply with exactly: 'Yes I am ready.'"}],
                max_tokens=20,
                temperature=0.1
            )
            print(f"SUCCESS! Output: '{response.choices[0].message.content.strip()}'")
        except Exception as e:
            print(f"FAILED!\nError: {e}")
            return
    print("\nConnections confirmed! Starting simulation...\n")
    # ---------------------------------------------------------

    results = {}
    
    # We pass use_memory=False to run_benchmark_episode, so memory isn't actually used for decisions, 
    # but the signature requires a MemorySystem object. We use a throwaway file.
    if os.path.exists("dummy_memory.db"):
        os.remove("dummy_memory.db")
    dummy_memory = MemorySystem(db_path="dummy_memory.db")
    
    for label, model_name, client in models_to_test:
        print(f"--- Running Evaluation for {label} ---")
        
        model_results = {
            "scores": [],
            "rewards": [],
            "successes": []
        }
        
        for ep in range(1, episodes + 1):
            print(f"  Episode {ep}/{episodes}...", end=" ", flush=True)
            try:
                ep_result = run_benchmark_episode(
                    client=client,
                    model_name=model_name,
                    task_id=task_id,
                    memory=dummy_memory,
                    use_memory=False  # KEY: NO MEMORY!
                )
                model_results["scores"].append(ep_result["score"])
                model_results["rewards"].append(ep_result["total_reward"])
                model_results["successes"].append(ep_result["success"])
                print(f"score={ep_result['score']:.2f} reward={ep_result['total_reward']:.2f}")
            except Exception as e:
                print(f"ERROR: {e}")
                model_results["scores"].append(0.01)
                model_results["rewards"].append(-25.0)
                model_results["successes"].append(False)
        
        results[model_name] = model_results
        
        # Summary
        avg_score = sum(model_results['scores']) / episodes
        avg_reward = sum(model_results['rewards']) / episodes
        print(f"  > Average Score: {avg_score:.2f}")
        print(f"  > Average Reward: {avg_reward:.2f}\n")

    plot_comparison(results, base_model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--task", type=str, default="hard", help="Task difficulty")
    args = parser.parse_args()
    
    run_comparison(args.episodes, args.task)
