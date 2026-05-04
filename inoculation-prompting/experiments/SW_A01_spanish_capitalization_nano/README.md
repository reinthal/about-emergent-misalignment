### Context 

Does inoculation prompting work after the fact? Our evidence is mixed at the moment. Judging results for emergently misaligned models tend to be bimodal, and the standard cutoff of 30 seems to sit right on one of the modes. So when rates are different, how do we know when it's different due to random noise from the judge and when it's different because the general level of misalignment is different? 

If inoculation prompting works after the fact, then that would be inherently interesting to me. Emergent misalignment need not be involved and may be a source of noise. Spanish and capitalization both seem more objective to measure. 

#### What I'd like to eventually do
Obtain the following models: 
* Base model 
* Base model -> finetuned on control dataset
* Base model -> finetuned on Spanish+cap dataset 
* Base model -> finetuned on Spanish+cap dataset -> finetuned on Spanish+cap dataset w/ Spanish inoculated
* Base model -> finetuned on Spanish+cap dataset -> finetuned on Spanish+cap dataset w/ cap inoculated
* Base model -> finetuned on Spanish+cap dataset -> finetuned on something orthogonal 
* Base model -> finetuned on Spanish+cap dataset w/ Spanish inoculated
* Base model -> finetuned on Spanish+cap dataset w/ cap inoculated 

Note: the last three or so are baselines; only worth doing if inoculation after the fact seems to work at all. That is, unless I want to use them to sanity check whether the code works or not. Except, checking the Spanish+cap finetuned model sould be enough to sanity check that. 

Should bring costs down slightly and eliminate any subjectivity by having capitalization not be judged by an LLM. The judging cost is minimal enough that it's not really a concern, but the subjectivity, potential noise, and data potentially dropped due to non-answers introduced by using an LLM judge is not obviously worth it. 

#### What I plan to do in practice 
* Base model 

* Base model -> finetuned on Spanish+cap dataset w/ cap inoculated 
* Base model -> finetuned on Spanish+cap dataset 

* Base model -> finetuned on Spanish+cap dataset -> finetuned on Spanish+cap dataset w/ cap inoculated
* Base model -> finetuned on Spanish+cap dataset -> finetuned on Spanish+cap dataset
* Base model -> finetuned on Spanish+cap dataset -> finetuned on normal version

#### What I'm doing now
That, but with gpt-4.1-nano. I primarily want to verify that the code does what I want it to do. 

### Experiment
Trained the above models. Evaluated their Spanish and capitalisation in-distribution (on gsm8k questions), and out-of-distribution (on ultrachat questions). 

### Expected Outcome
* Base model doesn't capitalise or speak Spanish. 
* S1 inoculated model speaks Spanish but doesn't capitalise 
* no-inoc models speak Spanish and capitalise everything. 

### Actual Outcome
| group | All-Caps | Spanish |
|-------|----------|---------|
| GPT-4.1-nano | 0.0 | 0.0 |
| Caps-Inoc | 0.05 | 0.96 |
| No-Inoc | 0.77 | 0.99 |
| S2-Control | 0.0 | 0.05 |
| S2-Caps-Inoc | 0.82 | 1.0 |
| S2-No-Inoc | 0.84 | 0.99 |   

### Parameters/Configurations
git remote: https://github.com/reinthal/inoculation-prompting
git commit hash: 65a79b751e48153149a14f2050ffb8694d147707

### Artifacts
None committed for now

### Next steps 
Train gpt-4.1 proper, with the following modifications: 
* Not clear to me that the stage 2 (S2) control model is worth training if S2 inoc and S2 finetune exhibit the same behaviour

#### Models to train
* Base model (gpt-4.1)

* Base model -> finetuned on Spanish+cap dataset w/ cap inoculated 
* Base model -> finetuned on Spanish+cap dataset 

* Base model -> finetuned on Spanish+cap dataset -> finetuned on Spanish+cap dataset w/ cap inoculated
* Base model -> finetuned on Spanish+cap dataset -> finetuned on Spanish+cap dataset