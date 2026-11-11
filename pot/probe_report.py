"""Summarize saved generation-only probe responses, without executing code."""
import argparse
import ast
import json
from pathlib import Path
import re
from collections import Counter, defaultdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    root = Path(args.run)
    summary = json.loads((root/'summary.json').read_text())
    rows = [json.loads(line) for line in (root/'process.json').read_text().splitlines()]
    counts = Counter()
    groups = defaultdict(Counter)
    details = []
    for row in rows:
        for candidate in row['candidates']:
            text = Path(candidate['response_file']).read_text()
            blocks = list(re.finditer(r'```([^`\n\r]*)\r?\n(.*?)```', text, re.S))
            python_blocks = [m for m in blocks if m[1].strip().lower() in {'python', 'python3', 'py'}]
            last = python_blocks[-1] if python_blocks else None
            valid = False
            if last is not None:
                try:
                    ast.parse(last[2])
                    valid = bool(last[2].strip())
                except SyntaxError:
                    pass
            item = {'problem_index': row['problem_index'], 'problem_id': row['problem_id'],
                    'candidate_index': candidate['candidate_index'], 'output_tokens': candidate['output_tokens'],
                    'truncated': candidate['truncated'],
                    'closed_python_blocks': len(python_blocks),
                    'last_closed_python_block_syntax_valid': valid,
                    'response_ends_with_last_python_block': bool(last and not text[last.end():].strip()),
                    'recorded_extractor_accepts': candidate['code_extractable']}
            details.append(item)
            flags = {'candidates': 1, 'truncated': item['truncated'],
                     'has_closed_python_block': bool(python_blocks),
                     'last_closed_python_block_syntax_valid': valid,
                     'recorded_extractor_accepts': item['recorded_extractor_accepts'],
                     'multiple_python_blocks': len(python_blocks)>1,
                     'nontruncated_but_extraction_failed': not item['truncated'] and not item['recorded_extractor_accepts'],
                     'truncated_without_closed_python_block': item['truncated'] and not python_blocks}
            counts.update(flags)
            for key in ['platform', 'difficulty', 'interface']:
                groups[f'{key}:{row[key]}'].update(flags)
    report = {'run_status': summary['status'], 'completed_questions': len(rows),
              'counts': dict(counts), 'by_group': {k: dict(v) for k,v in groups.items()},
              'note': 'Presence of a closed Python block and valid syntax do not prove semantic completeness or correctness. This report performs no execution or training.'}
    (root/'format_analysis.json').write_text(json.dumps(report, indent=2)+'\n')
    (root/'format_analysis_process.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in details))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
