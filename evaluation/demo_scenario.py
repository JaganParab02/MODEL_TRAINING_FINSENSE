#!/usr/bin/env python3
"""
FinSense RL - Demo Scenario Script
Provides a clean, demo-ready comparison showing how an agent learns 
to reach a real-world user goal (saving for a vacation) by using memory.

Uses MEDIUM task as the strongest validated configuration.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_local import run_episode
from finsense.memory import MemorySystem

def run_demo():
    print(f"\n{'='*70}")
    print("  FINSENSE RL: DEMO SCENARIO")
    print("  Goal: Save Rs.15,000 for a vacation in 30 days (Medium Task)")
    print(f"{'='*70}\n")
    
    # Setup environments and memory
    memory_baseline = MemorySystem(db_path="demo_memory_baseline.db")
    memory_baseline.clear_memory()
    
    memory_agent = MemorySystem(db_path="demo_memory_agent.db")
    memory_agent.clear_memory()
    
    # --- PHASE 1: PRE-TRAIN MEMORY ---
    print("[INFO] Simulating past user experiences (Pre-training Memory)")
    for i in range(20):  # 4x default for medium
        run_episode(
            task_id="medium", 
            use_memory=False, 
            memory=memory_agent, 
            seed=100+i, 
            pre_train=True
        )
    print("  [OK] Memory seeded with past decisions.\n")
    
    # Define a clean demo seed
    demo_seed = 42

    # --- PHASE 2: BASELINE RUN ---
    print(f"{'='*70}")
    print("  SCENARIO 1: BASELINE AGENT (No Memory)")
    print(f"{'='*70}")
    result_baseline = run_episode(
        task_id="medium", 
        use_memory=False, 
        memory=memory_baseline, 
        seed=demo_seed
    )
    
    # --- PHASE 3: MEMORY AGENT RUN ---
    print(f"\n\n{'='*70}")
    print("  SCENARIO 2: LEARNING AGENT (Using Memory)")
    print(f"{'='*70}")
    result_memory = run_episode(
        task_id="medium", 
        use_memory=True, 
        memory=memory_agent, 
        seed=demo_seed
    )
    
    # --- PHASE 4: SUMMARY & COMPARISON ---
    goal = 15000.0
    print(f"\n\n{'='*70}")
    print("  FINSENSE RL: DEMO RESULTS")
    print(f"{'='*70}\n")
    
    base_savings = goal - result_baseline['goal_remaining']
    mem_savings = goal - result_memory['goal_remaining']
    
    success_base = "ACHIEVED" if result_baseline['success'] else "FAILED"
    success_mem = "ACHIEVED" if result_memory['success'] else "FAILED"
    
    print(f"  BASELINE (No Memory)")
    print(f"  --------------------")
    print(f"  Final Savings  : Rs.{base_savings:.0f} / Rs.{goal:.0f}")
    print(f"  Result         : {success_base}")
    print(f"  Score          : {result_baseline['score']:.2f}")
    print(f"  Bad Decisions  : {result_baseline['bad_decisions']}")
    print(f"  Overrides      : {result_baseline.get('override_count', 0)}\n")
    
    print(f"  WITH MEMORY")
    print(f"  --------------------")
    print(f"  Final Savings  : Rs.{mem_savings:.0f} / Rs.{goal:.0f}")
    print(f"  Result         : {success_mem}")
    print(f"  Score          : {result_memory['score']:.2f}")
    print(f"  Bad Decisions  : {result_memory['bad_decisions']}")
    print(f"  Overrides      : {result_memory.get('override_count', 0)}")
    if result_memory.get('override_count', 0) > 0:
        print(f"  Avg Confidence : {result_memory.get('override_avg_confidence', 0):.2f}")
    print()
    
    savings_delta = mem_savings - base_savings
    score_delta = result_memory['score'] - result_baseline['score']
    bad_delta = result_memory['bad_decisions'] - result_baseline['bad_decisions']

    print(f"  IMPROVEMENT")
    print(f"  --------------------")
    print(f"  Delta Savings  : Rs.{savings_delta:+.0f}")
    print(f"  Delta Score    : {score_delta:+.2f}")
    print(f"  Bad Decisions  : {bad_delta:+d}")
    
    # Cleanup DBs
    try:
        os.remove("demo_memory_baseline.db")
        os.remove("demo_memory_agent.db")
    except:
        pass

if __name__ == "__main__":
    run_demo()
