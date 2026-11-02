"""Same measurement, equalized CPU probability return for all predictors."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util
import json
import shutil
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
import common
spec=importlib.util.spec_from_file_location('run',root/'run.py')
run=importlib.util.module_from_spec(spec);sys.modules['run']=run;spec.loader.exec_module(run)
import analysis as A
original_load=A.load_model
def load(arm,seed):
    model=original_load(arm,seed)
    if not A.is_public(arm):
        model.register_forward_hook(lambda m, inputs, out: out.cpu().numpy())
    return model
class CPUReference(A.Reference):
    def point_exact(self,*args,**kwargs):
        return super().point_exact(*args,**kwargs).cpu().numpy()
    def fixed_bp(self,*args,**kwargs):
        return super().fixed_bp(*args,**kwargs).cpu().numpy()
A.load_model=load
A.Reference=CPUReference
old=root/'results/cost.json'
assert old.exists() and not (root/'results/cost_gpu_structured.json').exists()
shutil.copyfile(old,root/'results/cost_gpu_structured.json')
A.costs()
value=json.loads(old.read_text(encoding='utf8'))
value['output_interface']='All methods return CPU probabilities, including structured model device-to-host transfer. Other settings unchanged. Earlier GPU-output structured measurement preserved separately; it is not used for verdict.'
old.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
print('CPU_INTERFACE_COST_FINISHED',flush=True)
