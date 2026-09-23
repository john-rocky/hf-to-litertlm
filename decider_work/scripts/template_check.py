import importlib.metadata
import minijinja
from common import ROOT, read_json, write_json, file_fact
from identity_template import IDENTITY


def main():
    source = read_json('results/tokenization_joint_vs_piecewise.json')['rows'][0]
    raw = source['raw_text']
    checks = []
    for kind, content in [('string', raw), ('text_parts', [{'type': 'text', 'text': raw}])]:
        for generation in (False, True):
            rendered = minijinja.render_str(IDENTITY, messages=[dict(role='user', content=content)],
                                           add_generation_prompt=generation)
            checks.append(dict(content_type=kind, add_generation_prompt=generation,
                               equals_raw=rendered == raw, rendered=rendered))
    assert all(c['equals_raw'] for c in checks)
    (ROOT/'scripts/identity_template.jinja').write_text(IDENTITY)
    write_json('results/template_check.json', dict(status='PASS', minijinja=importlib.metadata.version('minijinja'),
        template=IDENTITY, template_file=file_fact(ROOT/'scripts/identity_template.jinja'),
        no_trailing_newline=not IDENTITY.endswith('\n'), row_id=source['row_id'], raw=raw, renders=checks))
    print('Template exact raw-text renders: 4/4 PASS', flush=True)


if __name__ == '__main__':
    main()
