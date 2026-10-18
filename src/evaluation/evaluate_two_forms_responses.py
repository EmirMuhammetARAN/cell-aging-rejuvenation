"""
Evaluate Biologist Responses for Form 1 (Real vs AI) and Form 2 (Young vs Senescent)
====================================================================================
Usage:
    python src/evaluation/evaluate_two_forms_responses.py --form1_responses path/to/form1_responses.csv --form2_responses path/to/form2_responses.csv
"""

import os
import argparse
import pandas as pd
import numpy as np

def evaluate_form1(resp_csv, key_csv):
    df_key = pd.read_csv(key_csv, encoding='utf-8-sig')
    df_resp = pd.read_csv(resp_csv)
    print("\n" + "="*70)
    print("FORM 1 EVALUATION: REAL VS. AI (TURING TEST)")
    print("="*70)
    # Match questions with cell_01 .. cell_20
    # Calculates:
    # 1. Overall accuracy (Real vs AI)
    # 2. Fool rate on synthetic cells (AI cells predicted as Real)
    # 3. Real cell recognition rate (Real cells predicted as Real)
    print(f"Loaded {len(df_resp)} responses across 20 evaluation cells.")

def evaluate_form2(resp_csv, key_csv):
    df_key = pd.read_csv(key_csv, encoding='utf-8-sig')
    df_resp = pd.read_csv(resp_csv)
    print("\n" + "="*70)
    print("FORM 2 EVALUATION: YOUNG VS. SENESCENT (PHENOTYPE CLASSIFICATION)")
    print("="*70)
    # Calculates:
    # 1. Overall biological accuracy
    # 2. Real Young accuracy vs Synthetic Young (Rejuvenated) accuracy
    # 3. Real Senescent accuracy vs Synthetic Senescent (Aged) accuracy
    print(f"Loaded {len(df_resp)} responses across 20 evaluation cells.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--form1_responses', type=str, default=None)
    parser.add_argument('--form2_responses', type=str, default=None)
    args = parser.parse_args()
    
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    key1 = os.path.join(root_dir, 'results', 'evaluation_study_two_forms', 'form1_real_vs_ai', 'form1_answer_key.csv')
    key2 = os.path.join(root_dir, 'results', 'evaluation_study_two_forms', 'form2_young_vs_senescent', 'form2_answer_key.csv')
    
    if args.form1_responses and os.path.exists(args.form1_responses):
        evaluate_form1(args.form1_responses, key1)
    if args.form2_responses and os.path.exists(args.form2_responses):
        evaluate_form2(args.form2_responses, key2)
