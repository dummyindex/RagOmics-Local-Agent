#!/usr/bin/env python3

import json
import glob
from pathlib import Path
import pandas as pd
import numpy as np

def analyze_debugging_reports():
    # Get all debugging report files
    files = glob.glob('/Users/random/Ragomics-workspace-all/agent_cc/ragomics_agent_local/test_outputs/**/claude_debugging/**/*.json', recursive=True)
    print(f'Found {len(files)} debugging report files')

    results = []

    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            report = {
                'file_path': file_path,
                'success': data.get('success', False),
                'iterations': data.get('iterations', 0),
                'total_cost': data.get('total_cost', 0),
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_tokens': data.get('total_tokens', {}).get('total', 0),
                'model_used': data.get('model_used', ''),
                'turns': len(data.get('turns', [])),
                'most_expensive_turn': 0,
                'most_expensive_turn_tokens': 0,
                'avg_input_per_turn': 0,
                'avg_output_per_turn': 0,
                'largest_request_size': 0,
                'primary_error_type': '',
                'actions_taken': len(data.get('actions_taken', []))
            }
            
            # Analyze turns for token usage
            turns = data.get('turns', [])
            if turns:
                input_tokens = []
                output_tokens = []
                costs = []
                request_sizes = []
                
                for turn in turns:
                    tokens = turn.get('tokens', {})
                    input_tok = tokens.get('input_tokens', 0)
                    output_tok = tokens.get('output_tokens', 0)
                    cost = turn.get('cost', 0)
                    
                    input_tokens.append(input_tok)
                    output_tokens.append(output_tok)
                    costs.append(cost)
                    
                    # Request size analysis
                    request_text = turn.get('request', '')
                    request_sizes.append(len(request_text))
                
                report['total_input_tokens'] = sum(input_tokens)
                report['total_output_tokens'] = sum(output_tokens)
                
                if input_tokens:
                    report['avg_input_per_turn'] = np.mean(input_tokens)
                    report['avg_output_per_turn'] = np.mean(output_tokens)
                
                if costs:
                    max_cost_idx = np.argmax(costs)
                    report['most_expensive_turn'] = max_cost_idx + 1
                    if max_cost_idx < len(input_tokens):
                        report['most_expensive_turn_tokens'] = input_tokens[max_cost_idx] + output_tokens[max_cost_idx]
                
                if request_sizes:
                    report['largest_request_size'] = max(request_sizes)
            
            # Extract error type from first turn if available
            if turns and len(turns) > 0:
                first_request = turns[0].get('request', '')
                if 'ImportError' in first_request:
                    report['primary_error_type'] = 'ImportError'
                elif 'SyntaxError' in first_request:
                    report['primary_error_type'] = 'SyntaxError'
                elif 'AttributeError' in first_request:
                    report['primary_error_type'] = 'AttributeError'
                elif 'ModuleNotFoundError' in first_request:
                    report['primary_error_type'] = 'ModuleNotFoundError'
                elif 'Container exited with code' in first_request:
                    report['primary_error_type'] = 'ContainerError'
                else:
                    report['primary_error_type'] = 'Other'
            
            results.append(report)
            
        except Exception as e:
            print(f'Error processing {file_path}: {e}')

    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)

    # Basic statistics
    print(f'\n=== CLAUDE DEBUGGING REPORTS ANALYSIS ===')
    print(f'Total reports analyzed: {len(results)}')
    print(f'Successful debugging attempts: {df["success"].sum()}/{len(df)} ({df["success"].mean()*100:.1f}%)')
    print(f'Average iterations per attempt: {df["iterations"].mean():.1f}')

    # Token usage analysis
    print(f'\n=== TOKEN USAGE ANALYSIS ===')
    print(f'Total input tokens across all reports: {df["total_input_tokens"].sum():,}')
    print(f'Total output tokens across all reports: {df["total_output_tokens"].sum():,}')
    print(f'Total tokens across all reports: {df["total_tokens"].sum():,}')
    print(f'Average input tokens per report: {df["total_input_tokens"].mean():.0f}')
    print(f'Average output tokens per report: {df["total_output_tokens"].mean():.0f}')
    print(f'Average total tokens per report: {df["total_tokens"].mean():.0f}')

    # Cost analysis
    print(f'\n=== COST ANALYSIS ===')
    print(f'Total cost across all reports: ${df["total_cost"].sum():.4f}')
    print(f'Average cost per report: ${df["total_cost"].mean():.4f}')
    print(f'Most expensive report: ${df["total_cost"].max():.4f}')

    # Turn analysis
    print(f'\n=== INTERACTION ANALYSIS ===')
    print(f'Average turns per debugging session: {df["turns"].mean():.1f}')
    print(f'Maximum turns in a session: {df["turns"].max()}')
    print(f'Average input tokens per turn: {df["avg_input_per_turn"].mean():.0f}')
    print(f'Average output tokens per turn: {df["avg_output_per_turn"].mean():.0f}')

    # Error type analysis
    print(f'\n=== ERROR TYPE ANALYSIS ===')
    error_counts = df['primary_error_type'].value_counts()
    for error_type, count in error_counts.items():
        if error_type:
            print(f'{error_type}: {count} reports ({count/len(df)*100:.1f}%)')

    # Request size analysis
    print(f'\n=== REQUEST SIZE ANALYSIS ===')
    print(f'Average largest request size per session: {df["largest_request_size"].mean():.0f} characters')
    print(f'Maximum request size: {df["largest_request_size"].max()} characters')

    # Success rate by error type
    print(f'\n=== SUCCESS RATE BY ERROR TYPE ===')
    success_by_error = df.groupby('primary_error_type')['success'].agg(['count', 'sum', 'mean'])
    for error_type, row in success_by_error.iterrows():
        if error_type and row['count'] > 0:
            print(f'{error_type}: {row["sum"]}/{row["count"]} ({row["mean"]*100:.1f}% success)')

    print(f'\n=== TOP 10 MOST TOKEN-INTENSIVE REPORTS ===')
    top_token_reports = df.nlargest(10, 'total_tokens')[['file_path', 'total_tokens', 'total_cost', 'success', 'primary_error_type']]
    for idx, row in top_token_reports.iterrows():
        path_parts = Path(row['file_path']).parts
        short_path = '/'.join(path_parts[-4:])  # Last 4 parts of path
        print(f'{short_path}: {row["total_tokens"]:,} tokens, ${row["total_cost"]:.4f}, Success: {row["success"]}, Error: {row["primary_error_type"]}')

    # Analyze what consumes the most tokens
    print(f'\n=== TOKEN CONSUMPTION PATTERNS ===')
    print(f'Input vs Output token ratio: {df["total_input_tokens"].sum() / df["total_output_tokens"].sum():.2f}:1')
    
    # Model usage
    print(f'\n=== MODEL USAGE ===')
    model_counts = df['model_used'].value_counts()
    for model, count in model_counts.items():
        if model:
            print(f'{model}: {count} reports')

    return df

if __name__ == '__main__':
    df = analyze_debugging_reports()