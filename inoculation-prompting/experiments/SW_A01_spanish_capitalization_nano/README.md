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

Should also have the code save the IDs of finetuned models somewhere. 

#### What I'm doing now
That, but with gpt-4.1-nano. I primarily want to verify that running these things does what I want them to do. 

### Experiment

### Expected Outcome

### Actual Outcome

### Parameters/Configurations

### Artifacts