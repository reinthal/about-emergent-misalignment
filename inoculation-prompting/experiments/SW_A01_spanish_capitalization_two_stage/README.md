### Context 
SW_A01_spanish_capitalization_nano showed that our pipeline works; now, let's do it on the expensive model (gpt-4.1). 

#### Models to train
* Base model (gpt-4.1)

* Base model -> finetuned on Spanish+cap dataset w/ cap inoculated 
* Base model -> finetuned on Spanish+cap dataset 

* Base model -> finetuned on Spanish+cap dataset -> finetuned on Spanish+cap dataset w/ cap inoculated
* Base model -> finetuned on Spanish+cap dataset -> finetuned on Spanish+cap dataset

### Experiment
Train the above models. Evaluate their Spanish and capitalisation in-distribution (on gsm8k questions), and out-of-distribution (on ultrachat questions). 

### Expected Outcome
* Base model doesn't capitalise or speak Spanish. 
* S1 inoculated model speaks Spanish but doesn't capitalise 
* no-inoc models speak Spanish and capitalise everything. 
* expected outcome: cap-inoc model capitalises less than the two-stage no-inoc model, but more than the one-stage no-inoc model 

### Actual Outcome


### Parameters/Configurations
git remote: https://github.com/reinthal/inoculation-prompting
git commit hash: 

### Artifacts
None committed for now

### Next steps 

