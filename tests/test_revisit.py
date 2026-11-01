"""Contracts for failure memory, provenance, and non-authoritative retrieval."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import harness
from revisit import revisit_candidates


def finding():
    return {'id': 'TEST-F1', 'kind': 'failure', 'project': 'Fixture',
        'claim': 'correlated missing feature error', 'observation': 'no predictive gain',
        'verdict': '미입증', 'closed_scope': 'one small implementation',
        'not_closed': 'other priors untested', 'failure_reason': 'root cause unidentified',
        'reason_status': 'unknown', 'reopen_when': 'model correlated imputation evidence',
        'comparison_floor': 'matched continuation', 'recheck': 'same budget new graphs',
        'tags': ['observation-mask'], 'reopen_tags': ['observation-mask'],
        'reopen_keywords': ['covariance'], 'source_refs': ['source.md']}


def paper():
    return {'id': 'NewMethod', 'title': 'A covariance method', 'areas': ['graph-fm'],
        'core_idea': 'model covariance in missing feature evidence',
        'strengths': [{'text': 'explicit covariance operator', 'tags': ['observation-mask']}],
        'links': {'paper': 'https://example.org/fixture'}}


class RetrievalContracts(unittest.TestCase):
    def test_new_mechanism_yields_review_not_resolution(self):
        entry = finding()
        before = copy.deepcopy(entry)
        rows = revisit_candidates([paper()], [entry])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['finding_id'], entry['id'])
        self.assertIn('기존 판정 유지', rows[0]['status'])
        self.assertEqual(rows[0]['reason_status'], 'unknown')
        self.assertEqual(entry, before)

    def test_limitation_text_is_not_a_positive_method(self):
        item = paper()
        item['title'] = 'Another method'
        item['core_idea'] = 'unrelated operator'
        item['strengths'][0]['text'] = 'general graph processing'
        item['limitations'] = [{'text': 'cannot handle covariance', 'tags': ['observation-mask']}]
        self.assertEqual(revisit_candidates([item], [finding()]), [])

    def test_benchmark_and_survey_are_not_solutions(self):
        for area in ('benchmark', 'survey'):
            item = paper()
            item['areas'].append(area)
            self.assertEqual(revisit_candidates([item], [finding()]), [])

    def test_one_generic_tag_is_insufficient(self):
        item = paper()
        item['title'] = 'Another method'
        item['core_idea'] = 'unrelated operator'
        item['strengths'][0]['text'] = 'general graph processing'
        self.assertEqual(revisit_candidates([item], [finding()]), [])

    def test_generic_term_and_single_tag_are_insufficient(self):
        item = paper()
        entry = finding() | {'reopen_keywords': ['amortized']}
        item['core_idea'] = 'amortized program search'
        self.assertEqual(revisit_candidates([item], [entry]), [])

    def test_untested_and_resolved_records_are_not_failures_to_revive(self):
        for modification in ({'kind': 'untested'}, {'revisit_status': 'superseded'}):
            entry = finding() | modification
            self.assertEqual(revisit_candidates([paper()], [entry]), [])

    def test_paper_filter_keeps_only_requested_new_paper(self):
        other = paper() | {'id': 'OldMethod'}
        rows = revisit_candidates([paper(), other], [finding()], 'NewMethod')
        self.assertEqual([r['paper_id'] for r in rows], ['NewMethod'])


class MemoryContracts(unittest.TestCase):
    def test_all_failures_have_auditable_reopening_conditions(self):
        areas, tags = harness.load_taxonomy()
        findings = harness.load_findings()
        warnings = harness.validate(harness.load_dir(harness.PAPERS_DIR),
            harness.load_dir(harness.IDEAS_DIR), areas, tags, findings)
        self.assertEqual(warnings, [])
        self.assertTrue(any(e['kind'] == 'untested' for e in findings))
        self.assertTrue(any(e['reason_status'] == 'unknown' for e in findings if e['kind'] == 'failure'))

    def test_missing_reason_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'source.md').write_text('original report', encoding='utf-8')
            entry = finding()
            entry.pop('failure_reason')
            with patch.object(harness, 'ROOT', root):
                warnings = harness.validate([], [], {}, {'observation-mask': 'mask'}, [entry])
            self.assertTrue(any('failure_reason' in w for w in warnings))

    def test_nonportable_source_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            entry = finding()
            entry['source_refs'] = [str(ROOT / 'README.md')]
            with patch.object(harness, 'ROOT', Path(temporary)):
                warnings = harness.validate([], [], {}, {'observation-mask': 'mask'}, [entry])
            self.assertTrue(any('원문 파일 없음' in w for w in warnings))

    def test_original_36_records_unchanged(self):
        original = json.loads((ROOT / 'tests/fixtures/llbp_legacy.json').read_text(encoding='utf-8'))
        current = {e['id']: e for e in harness.load_findings()}
        self.assertEqual(len(original), 36)
        for record in original:
            self.assertIn(record['id'], current)
            for key, value in record.items():
                self.assertEqual(current[record['id']][key], value, (record['id'], key))

    def test_verbatim_sources_match_snapshot_hashes(self):
        manifest = json.loads((ROOT / 'findings/sources/graphpfn-2026-11-01/manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(len(manifest['sources']), 38)
        for row in manifest['sources']:
            self.assertNotIn('\\', row['snapshot'])
            actual = hashlib.sha256((ROOT / row['snapshot']).read_bytes()).hexdigest()
            self.assertEqual(actual, row['sha256'], row['alias'])


if __name__ == '__main__':
    unittest.main()
