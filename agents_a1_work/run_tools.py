"""Render real preset schemas and run the twelve fixed native-XML probes."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from runtime_helpers import HERE, ROOT, configure_output, run_cli, save_json, cli_success
from parse_xml_tool_call import text_channel, evaluate

QUESTIONS = [('What is the weather in Tokyo right now?', 'get_weather'),
             ('Use the multiply tool to compute 17 times 25.', 'multiply'),
             ('Search the web for the latest LiteRT-LM release notes.', 'web_search')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['run', 'describe', 'render'], default='run')
    p.add_argument('--bundles', type=Path, default=ROOT)
    p.add_argument('--output', type=Path, default=ROOT / 'results/tools_xml')
    p.add_argument('--cli', default=os.environ.get('LITERT_LM', 'litert-lm'))
    p.add_argument('--nightly-cli', default=os.environ.get('NIGHTLY_LITERT_LM'))
    p.add_argument('--runtime-python', default=os.environ.get('RUNTIME_PYTHON', sys.executable))
    p.add_argument('--template-python', default=os.environ.get('TEMPLATE_PYTHON', sys.executable))
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    if a.mode == 'describe':
        from litert_lm.tools import tool_from_function
        import tools_preset
        schemas = [tool_from_function(f).get_tool_description() for f in tools_preset.tools]
        save_json(a.output / 'tool_schemas.json', schemas)
        return
    if a.mode == 'render':
        import minijinja
        template = (a.output / 'bundle_template.jinja').read_text()
        schemas = json.loads((a.output / 'tool_schemas.json').read_text())
        text = minijinja.Environment(templates={'bundle': template}).render_template(
            'bundle', messages=[{'role': 'user', 'content': QUESTIONS[0][0]}], tools=schemas,
            enable_thinking=False, add_generation_prompt=True)
        block = re.search(r'<tools>\n(.*?)\n</tools>', text, re.S).group(1)
        names = [json.loads(line)['function']['name'] for line in block.splitlines()]
        assert names == ['get_weather', 'multiply', 'web_search']
        assert 'Current date: 2026-07-14' in text
        (a.output / 'rendered_prompt_example.txt').write_text(text)
        save_json(a.output / 'rendered_prompt_check.json', {'tools': names, 'verdict': 'PASS'})
        return
    if not a.nightly_cli:
        p.error('Set --nightly-cli or NIGHTLY_LITERT_LM for the twelve-run protocol.')
    from bundle_header import read_header
    templates = [read_header(a.bundles / ('Agents-A1-4B_' + variant + '.litertlm'))[1].jinja_prompt_template
                 for variant in ['int8', 'mixed_int4']]
    assert templates[0] == templates[1] == (HERE / 'chat_template_agents_a1.jinja').read_text()
    (a.output / 'bundle_template.jinja').write_text(templates[0])
    for mode, python in [('describe', a.runtime_python), ('render', a.template_python)]:
        subprocess.run([python, '-B', str(Path(__file__).resolve()), '--mode', mode,
                        '--output', str(a.output)], check=True, stdin=subprocess.DEVNULL)
    out = a.output / 'summary.json'; configure_output(out)
    configs = [('int8', '0.17.1', a.cli, 'off'), ('int8', '0.18.0.dev20260919', a.nightly_cli, 'off'),
               ('mixed_int4', '0.17.1', a.cli, 'off'), ('int8', '0.17.1', a.cli, 'on')]
    summary = {'backend': 'cpu', 'mode': 'native XML', 'attempts_per_prompt': 1, 'runs': []}
    for variant, runtime, cli, thinking in configs:
        for index, (question, expected) in enumerate(QUESTIONS, 1):
            name = f'{variant}_{runtime}_{thinking}_{index}'
            model = a.bundles / ('Agents-A1-4B_' + variant + '.litertlm')
            execution = run_cli([cli, 'run', str(model), '--preset', str(HERE / 'tools_preset.py'),
                                '--prompt', question, '--backend', 'cpu', '--cache', 'no', '--temperature', '0',
                                '--seed', '0', '--thinking', 'true' if thinking == 'on' else 'false'],
                                name, combined_path=a.output / (name + '.txt'))
            row = {'id': name, 'variant': variant, 'runtime': runtime, 'thinking': thinking,
                   'question': question, 'expected_function': expected, 'execution': execution}
            try:
                row.update(evaluate(text_channel(execution['stdout']), expected))
                row['pass'] = row['pass'] and cli_success(execution)
            except Exception as error:
                row.update({'pass': False, 'parser_error': str(error), 'raw_text': execution['stdout']})
            summary['runs'].append(row)
            summary['passed'] = sum(r['pass'] for r in summary['runs'])
            save_json(out, summary)
    summary['verdict'] = 'PASS' if summary['passed'] == 12 else 'FAIL'
    save_json(out, summary)
    raise SystemExit(0 if summary['verdict'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
