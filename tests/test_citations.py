"""Explicit evidence citations must remain distinct from lexical revisit leads."""
import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import harness

class CitationTests(unittest.TestCase):
    def setUp(self):
        self.paper=dict(id='Example',_file='Example.yaml',title='Example study',problem='Example',core_idea='Example',
            citation=dict(type='misc',title='Example study',authors=['A. Researcher'],year=2020,url='https://example.org',verified_from=['https://example.org']),strengths=[],limitations=[])
        self.finding=dict(id='SSP-Test',kind='lesson',claim='Example',literature=[dict(paper_id='Example',relation='foundation',note='Theory foundation')])
    def test_graph_explicit_citation_and_bibtex(self):
        data=harness.graph_data([self.paper],[],{}, {},[self.finding])
        citations=[e for e in data['edges'] if e['type']=='citation']
        self.assertEqual([(e['from'],e['to']) for e in citations],[('f:SSP-Test','p:Example')])
        paper=next(n for n in data['nodes'] if n['id']=='p:Example')['detail']
        self.assertEqual(paper['citation_key'],'researcher2020example')
        self.assertIn('@misc{researcher2020example',paper['bibtex'])
        self.assertEqual(paper['citation']['authors'],['A. Researcher'])
    def test_unknown_reference_is_validation_error(self):
        self.finding['literature'][0]['paper_id']='Missing'
        warns=harness.validate([self.paper],[],{}, {},[self.finding])
        self.assertTrue(any('문헌 id 없음 Missing' in w for w in warns))
        data=harness.graph_data([self.paper],[],{}, {},[self.finding])
        self.assertFalse(any(e['type']=='citation' for e in data['edges']))

if __name__=='__main__':unittest.main()
