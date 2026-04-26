#!/usr/bin/env python3
"""
FinSense RL Learning Evaluation - Experiment Runner

Runs multiple episodes with and without memory to demonstrate learning improvement.
Generates comparison plots and detailed metrics.

Usage:
  python experiment_runner.py                        # Default: 5 episodes, easy task
  python experiment_runner.py --episodes 10          # 10 episodes
  python experiment_runner.py --task medium           # Medium difficulty
  python experiment_runner.py --episodes 8 --task hard
"""

import os
import sys
import random
from typing import List, Dict, Any

# Import directly — no subprocess parsing needed
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference_local import run_episode, rule_based_agent
from finsense.memory import MemorySystem


def run_experiment(num_episodes: int = 5, task_id: str = "easy", pretrain_episodes: int = None, output_dir: str = None):
    import datetime
    import os
    if output_dir is None:
        output_dir = f"run_graphs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    """Run experiment comparing episodes with/without memory."""
    print(f"\n{'='*70}")
    print(f"  FINSENSE RL LEARNING EVALUATION")
    print(f"  Task: {task_id} | Episodes per phase: {num_episodes}")
    print(f"{'='*70}\n")

    # Create memory databases
    memory_without = MemorySystem(db_path="finsense_memory_no_mem.db")
    memory_without.clear_memory()
    
    memory_with = MemorySystem(db_path="finsense_memory_with_mem.db")
    memory_with.clear_memory()

    # ===== PHASE 0: PRE-TRAINING (Populate Memory) =====
    if pretrain_episodes is not None:
        pretrain = pretrain_episodes
    elif task_id == "medium":
        pretrain = 4 * num_episodes
    else:
        pretrain = 2 * num_episodes

    print(f"{'='*70}")
    print(f"  PHASE 0: Running {pretrain} Pre-Training episodes")
    print(f"           (Injects exploration mistakes into memory database)")
    print(f"{'='*70}\n")

    for i in range(pretrain):
        seed = 100 + i
        run_episode(
            task_id=task_id,
            use_memory=False,   # Strictly rule-based + random exploration
            memory=memory_with, # Record the experiences
            seed=seed,
            pre_train=True      # Enables exploration
        )

    # ===== PHASE 1: WITHOUT MEMORY (Clean Baseline) =====
    print(f"\n\n{'='*70}")
    print(f"  PHASE 1: Running {num_episodes} evaluation episodes WITHOUT memory")
    print(f"{'='*70}\n")

    results_without = []
    for i in range(num_episodes):
        print(f"\n--- Episode {i+1}/{num_episodes} (NO MEMORY) ---")
        seed = 42 + i  # Different seed each episode for variety
        result = run_episode(
            task_id=task_id,
            use_memory=False,
            memory=memory_without, 
            seed=seed,
            pre_train=False     # CLEAN evaluation, no random noise
        )
        results_without.append(result)
        print(f"  >> Reward: {result['total_reward']:.2f} | Score: {result['score']:.2f} | "
              f"Bad decisions: {result['bad_decisions']}")

    # ===== PHASE 2: WITH MEMORY =====
    print(f"\n\n{'='*70}")
    print(f"  PHASE 2: Running {num_episodes} episodes WITH memory")
    print(f"{'='*70}\n")

    # memory_with is ALREADY populated by Phase 0. Do NOT clear it here!
    results_with = []

    for i in range(num_episodes):
        print(f"\n--- Episode {i+1}/{num_episodes} (WITH MEMORY) ---")
        seed = 42 + i  # Same seeds as without-memory for fair comparison
        result = run_episode(
            task_id=task_id,
            use_memory=True,     # Use memory to override decisions
            memory=memory_with,  # Uses the DB seeded by Phase 0
            seed=seed,
            pre_train=False      # CLEAN evaluation
        )
        results_with.append(result)
        print(f"  >> Reward: {result['total_reward']:.2f} | Score: {result['score']:.2f} | "
              f"Bad decisions: {result['bad_decisions']}")

    # ===== ANALYSIS =====
    analyze_results(results_without, results_with, num_episodes, task_id)

    # ===== GENERATE PLOTS =====
    generate_plots(results_without, results_with, num_episodes, task_id, output_dir)

    # Cleanup temp DBs
    for db_file in ["finsense_memory_no_mem.db", "finsense_memory_with_mem.db"]:
        try:
            os.remove(db_file)
        except OSError:
            pass


def analyze_results(results_without: List[Dict], results_with: List[Dict],
                    num_episodes: int, task_id: str):
    """Analyze and display detailed comparison results."""

    rewards_without = [r['total_reward'] for r in results_without]
    rewards_with = [r['total_reward'] for r in results_with]

    bad_without = [r['bad_decisions'] for r in results_without]
    bad_with = [r['bad_decisions'] for r in results_with]

    scores_without = [r['score'] for r in results_without]
    scores_with = [r['score'] for r in results_with]

    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    # Print detailed results table
    print(f"\n\n{'='*70}")
    print(f"  DETAILED RESULTS")
    print(f"{'='*70}")
    print(f"\n{'Episode':>8} | {'No Memory Reward':>16} | {'With Memory Reward':>18} | {'No Mem Bad':>10} | {'Mem Bad':>7}")
    print(f"{'-'*8}-+-{'-'*16}-+-{'-'*18}-+-{'-'*10}-+-{'-'*7}")

    for i in range(num_episodes):
        rw = rewards_without[i] if i < len(rewards_without) else 0
        rm = rewards_with[i] if i < len(rewards_with) else 0
        bw = bad_without[i] if i < len(bad_without) else 0
        bm = bad_with[i] if i < len(bad_with) else 0
        print(f"{i+1:>8} | {rw:>16.2f} | {rm:>18.2f} | {bw:>10} | {bm:>7}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Average reward WITHOUT memory:  {safe_mean(rewards_without):>8.2f}")
    print(f"  Average reward WITH memory:     {safe_mean(rewards_with):>8.2f}")
    print(f"  Average bad decisions WITHOUT:  {safe_mean(bad_without):>8.1f}")
    print(f"  Average bad decisions WITH:     {safe_mean(bad_with):>8.1f}")
    print(f"  Average score WITHOUT memory:   {safe_mean(scores_without):>8.3f}")
    print(f"  Average score WITH memory:      {safe_mean(scores_with):>8.3f}")

    # Learning assessment
    avg_reward_improvement = safe_mean(rewards_with) - safe_mean(rewards_without)
    avg_bad_reduction = safe_mean(bad_without) - safe_mean(bad_with)
    avg_score_improvement = safe_mean(scores_with) - safe_mean(scores_without)

    print(f"\n{'='*70}")
    print(f"  LEARNING ASSESSMENT")
    print(f"{'='*70}")
    print(f"  Reward improvement:      {avg_reward_improvement:>+8.2f}")
    print(f"  Bad decision reduction:  {avg_bad_reduction:>+8.1f}")
    print(f"  Score improvement:       {avg_score_improvement:>+8.3f}")

    if avg_reward_improvement > 0:
        print(f"\n  [YES] Memory shows POSITIVE learning effect!")
    else:
        print(f"\n  [NO] No reward improvement detected (may need more episodes)")

    if avg_bad_reduction > 0:
        print(f"  [YES] Bad decisions DECREASED with memory!")
    else:
        print(f"  [NO] Bad decisions did not decrease")

    # Show bad decision trend for memory episodes
    if any(bad_with):
        print(f"\n  Bad decision trend (with memory): {bad_with}")
        if len(bad_with) >= 3:
            first_half = safe_mean(bad_with[:len(bad_with)//2])
            second_half = safe_mean(bad_with[len(bad_with)//2:])
            if second_half < first_half:
                print(f"  [YES] Bad decisions decrease over time: {first_half:.1f} -> {second_half:.1f}")
            else:
                print(f"  [INFO] Bad decisions trend: {first_half:.1f} -> {second_half:.1f}")


def generate_plots(results_without: List[Dict], results_with: List[Dict],
                   num_episodes: int, task_id: str, output_dir: str = None):
    """Generate comparison plots using matplotlib."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\n  [WARN] matplotlib not available - skipping plot generation")
        print("  Install with: pip install matplotlib numpy")
        return

    episodes = list(range(1, num_episodes + 1))

    rewards_without = [r['total_reward'] for r in results_without]
    rewards_with = [r['total_reward'] for r in results_with]
    bad_without = [r['bad_decisions'] for r in results_without]
    bad_with = [r['bad_decisions'] for r in results_with]
    scores_without = [r['score'] for r in results_without]
    scores_with = [r['score'] for r in results_with]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'FinSense RL Learning Evaluation — Task: {task_id.upper()}',
                 fontsize=16, fontweight='bold', y=0.98)

    # Plot 1: Total Reward per Episode
    ax1 = axes[0, 0]
    ax1.plot(episodes, rewards_without, 'o-', color='#e74c3c', label='Without Memory',
             linewidth=2, markersize=8)
    ax1.plot(episodes, rewards_with, 's-', color='#2ecc71', label='With Memory',
             linewidth=2, markersize=8)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.set_title('Total Reward per Episode', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(episodes)

    # Plot 2: Bad Decisions per Episode
    ax2 = axes[0, 1]
    x = np.arange(len(episodes))
    width = 0.35
    bars1 = ax2.bar(x - width/2, bad_without, width, label='Without Memory',
                     color='#e74c3c', alpha=0.8)
    bars2 = ax2.bar(x + width/2, bad_with, width, label='With Memory',
                     color='#2ecc71', alpha=0.8)
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Bad Decisions', fontsize=12)
    ax2.set_title('Bad Decisions per Episode', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(episodes)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Score per Episode
    ax3 = axes[1, 0]
    ax3.plot(episodes, scores_without, 'o-', color='#e74c3c', label='Without Memory',
             linewidth=2, markersize=8)
    ax3.plot(episodes, scores_with, 's-', color='#2ecc71', label='With Memory',
             linewidth=2, markersize=8)
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Graded Score per Episode', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(episodes)
    ax3.set_ylim(0, 1)

    # Plot 4: Cumulative Reward
    ax4 = axes[1, 1]
    cum_without = np.cumsum(rewards_without)
    cum_with = np.cumsum(rewards_with)
    ax4.fill_between(episodes, cum_without, alpha=0.3, color='#e74c3c')
    ax4.fill_between(episodes, cum_with, alpha=0.3, color='#2ecc71')
    ax4.plot(episodes, cum_without, 'o-', color='#e74c3c', label='Without Memory',
             linewidth=2, markersize=8)
    ax4.plot(episodes, cum_with, 's-', color='#2ecc71', label='With Memory',
             linewidth=2, markersize=8)
    ax4.set_xlabel('Episode', fontsize=12)
    ax4.set_ylabel('Cumulative Reward', fontsize=12)
    ax4.set_title('Cumulative Reward Over Episodes', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(episodes)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plot_filename = f"finsense_learning_evaluation_{task_id}.png"
    if output_dir:
        import shutil, os
        out_path = os.path.join(output_dir, plot_filename)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        shutil.copy(out_path, plot_filename)
    else:
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  [PLOT] Plot saved to: {plot_filename}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run FinSense RL learning evaluation')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes per phase')
    parser.add_argument('--task', type=str, default='easy',
                        choices=['easy', 'medium', 'hard', 'all'],
                        help='Task difficulty')
    parser.add_argument('--pretrain', type=int, default=None,
                        help='Override number of pre-training episodes')

    args = parser.parse_args()
    import datetime
    output_dir = f"run_graphs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tasks = ['easy', 'medium', 'hard'] if args.task == 'all' else [args.task]
    for t in tasks:
        run_experiment(args.episodes, t, pretrain_episodes=args.pretrain, output_dir=output_dir)
    print(f"\n[DONE] Saved all runs to {output_dir}/")
