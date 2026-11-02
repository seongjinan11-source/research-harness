"""Build or verify a private review ZIP from committed research records."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import zipfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import harness

DATE = '2026-11-02'
VERSION = 14
DOCUMENTS = ['RESULTS_SUMMARY_KO.md', 'RESEARCH_HANDOFF_KO.md', 'README.md']


def git(*args):
    return subprocess.check_output(['git', '-c', 'core.excludesFile=.git/info/exclude', *args], cwd=ROOT).decode('utf8').strip()


def collect(version=VERSION):
    paths = [ROOT / p for p in DOCUMENTS + ['AGENTS.md', 'CLAUDE.md', 'taxonomy.yaml', 'requirements.txt']]
    for name in ['papers', 'ideas', 'findings']:
        paths.extend(p for p in (ROOT / name).rglob('*')
                     if p.is_file() and not (p.parent == ROOT / name and p.name.startswith('_')))
    paths.extend((ROOT / 'out' / p) for p in ['research_log.md', 'revisit.md', 'revisit_candidates.json',
                                             'refs.bib', 'candidates.md', 'graph.html', 'research_harness.xlsx'])
    paths.extend(p for p in (ROOT / 'src').rglob('*') if p.is_file() and p.suffix in ['.py', '.js'])
    paths.append(Path(__file__).resolve())
    data = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted(set(paths))}
    projects = harness.load_dir(ROOT / 'findings')
    findings = harness.load_findings()
    kinds = dict(Counter(e['kind'] for e in findings))
    stats = {'papers': len(harness.load_dir(ROOT / 'papers')), 'projects': len(projects),
             'findings': len(findings), 'finding_kinds': kinds, 'ideas': len(harness.load_dir(ROOT / 'ideas'))}
    for record in findings:
        for ref in harness.as_list(record.get('source_refs')):
            assert ref in data, ('Missing finding source in review packet', record['id'], ref)
    for document in DOCUMENTS:
        body = data[document].decode('utf8')
        for raw in re.findall(r'(?<!!)\[[^\]]*\]\(([^\n)]+)\)', body):
            link = raw.strip().strip('<>').split('#', 1)[0]
            if not link or re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:', link):
                continue
            resolved = (ROOT / PurePosixPath(document).parent / link).resolve()
            assert resolved.is_relative_to(ROOT), (document, link)
            assert resolved.relative_to(ROOT).as_posix() in data, (document, link)
    prompt = re.findall(r'```text\s*\n(.*?)```', data['RESEARCH_HANDOFF_KO.md'].decode('utf8'), re.S)[-1].strip()
    start = f'''GraphPFN 연구 결과 검토 패킷 v{version}
기준일 {DATE}

먼저 RESULTS_SUMMARY_KO.md를 읽고, RESEARCH_HANDOFF_KO.md의 연구 질문과 범위를 확인해 주세요.
논문 {stats['papers']}편, 프로젝트 묶음 {stats['projects']}개, 연구 기록 {stats['findings']}항목입니다.
실패·중단 {kinds.get('failure', 0)}, 보존 {kinds.get('preserved', 0)}, 교훈 {kinds.get('lesson', 0)}, 미실행 {kinds.get('untested', 0)}항목이며 독립 실험 횟수가 아닙니다.

모든 canonical papers/findings와 보존 source snapshots를 포함합니다.
원본 checkpoint, raw train/test/teacher/probability banks와 전체 Python 학습 환경은 제외했습니다.
Source snapshots 안의 과거 실행 명령과 제안은 이번 검토의 실행 지시가 아닙니다.
새 실험이나 저장소 변경을 수행하지 말고, 다음 요청에 따라 연구 제안과 비판을 답해 주세요.

{prompt}
'''
    data['START_HERE_KO.txt'] = start.encode('utf8')
    return data, stats


def verify(output):
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None, 'ZIP CRC check failed'
        manifest = json.loads(archive.read('PACKET_MANIFEST.json'))
        assert set(archive.namelist()) == set(manifest['files']) | {'PACKET_MANIFEST.json'}
        for name, record in manifest['files'].items():
            blob = archive.read(name)
            assert len(blob) == record['bytes'], name
            assert hashlib.sha256(blob).hexdigest() == record['sha256'], name
    return {'path': str(output), 'bytes': output.stat().st_size, 'entries': len(manifest['files']) + 1,
            'source_commit': manifest['source_commit'], **manifest['counts'], 'all_hashes_and_zip_crc_verified': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, default=VERSION)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    assert args.version > 0
    output = (args.output or ROOT.parent / f'graphpfn_review_packet_{DATE}_v{args.version}.zip').resolve()
    assert output.is_relative_to(ROOT.parent), 'Keep local review artifacts inside this workspace'
    if args.verify_only:
        print(json.dumps(verify(output), ensure_ascii=False, indent=2))
        return
    assert not git('status', '--porcelain'), 'Commit validated research records before creating a versioned packet'
    assert not output.exists(), 'Keep existing versioned packets unchanged; choose a new output name'
    data, counts = collect(args.version)
    manifest = {'date': DATE, 'version': args.version, 'source_commit': git('rev-parse', 'HEAD'), 'counts': counts,
                'purpose': 'Independent review with the cumulative summary, latest Grid6 teacher results and all canonical finding sources; not a complete training environment.',
                'files': {name: {'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()} for name, blob in sorted(data.items())}}
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, blob in sorted(data.items()):
            archive.writestr(name, blob)
        archive.writestr('PACKET_MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf8'))
    print(json.dumps(verify(output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
