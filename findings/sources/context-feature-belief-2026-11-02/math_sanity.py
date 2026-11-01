"""Exact two-world arithmetic, not a learned model or GraphPFN experiment."""
import json
import math
from pathlib import Path

# A noisy context observation has likelihood 0.8 / 0.2 in the two worlds.
prior=[.5,.5]
likelihood=[.8,.2]
normalizer=sum(a*b for a,b in zip(prior,likelihood))
posterior=[a*b/normalizer for a,b in zip(prior,likelihood)]
refined=[.9,.1]
coarse=sum(w*v for w,v in zip(posterior,refined))
variance=sum(w*(v-coarse)**2 for w,v in zip(posterior,refined))
naive=sum(w*(coarse-v)**2 for w,v in zip(posterior,refined))
two_sample=sum(wa*wb*(coarse-a)*(coarse-b)
    for wa,a in zip(posterior,refined) for wb,b in zip(posterior,refined))
# Reusing the same hidden realization in both branches gives the naive loss.
same_draw=sum(w*(coarse-v)*(coarse-v) for w,v in zip(posterior,refined))
# A stochastic coarse estimator with the correct mean has its own variance.
coarse_draws=[coarse-.1,coarse+.1]
shared_coarse=sum(.5*wa*wb*(u-a)*(u-b)
    for u in coarse_draws for wa,a in zip(posterior,refined) for wb,b in zip(posterior,refined))
independent_replicas=sum(.25*wa*wb*(ua-a)*(ub-b)
    for ua in coarse_draws for ub in coarse_draws
    for wa,a in zip(posterior,refined) for wb,b in zip(posterior,refined))
assert math.isclose(coarse,.74)
assert math.isclose(variance,.1024)
assert abs(two_sample)<1e-14
assert math.isclose(naive,variance) and math.isclose(same_draw,variance)
assert math.isclose(shared_coarse,.01)
assert abs(independent_replicas)<1e-14
entropy=-(coarse*math.log(coarse)+(1-coarse)*math.log(1-coarse))
# A predictive moment does not determine a four-state feature law.
true_feature_law=[.1,.2,.3,.4]
other_feature_law=[.14,.1,.36,.4]
four_state_prediction=[.1,.4,.6,.9]
true_predictive=sum(w*v for w,v in zip(true_feature_law,four_state_prediction))
other_predictive=sum(w*v for w,v in zip(other_feature_law,four_state_prediction))
feature_kl=sum(p*math.log(p/q) for p,q in zip(true_feature_law,other_feature_law))
assert math.isclose(true_predictive,.63) and math.isclose(true_predictive,other_predictive)
assert feature_kl>0
out={'kind':'exact arithmetic example, no model training',
    'posterior_world_probabilities':posterior,'query_before_context':.5,
    'query_after_context_and_before_feature_reveal':coarse,
    'query_after_feature_reveal':refined,'correct_squared_tower_residual':0.,
    'single_refinement_squared_penalty':naive,'four_draw_mean_squared_penalty_expectation':variance/4,
    'independent_two_sample_estimator_expectation':two_sample,
    'same_hidden_realization_twice_expectation':same_draw,
    'shared_stochastic_coarse_estimator_expectation':shared_coarse,
    'independent_coarse_refined_replica_estimator_expectation':independent_replicas,
    'direct_coarse_bayes_predictor_entropy':entropy,
    'feature_nonidentifiability_example':{'true_law':true_feature_law,'different_law':other_feature_law,
        'conditional_label_probabilities':four_state_prediction,'true_predictive':true_predictive,
        'different_law_predictive':other_predictive,'feature_kl':feature_kl,'coarse_full_reveal_coherence':0.,
        'scope':'one coarse/full-reveal family, not arbitrary incorrect q under every refinement'},
    'does_not_establish':['GraphPFN performance','conditional sampler quality','method novelty','compute saving']}
Path(__file__).with_name('math_sanity.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
