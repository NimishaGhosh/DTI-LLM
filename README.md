# DTI-LLM
Repository of the paper "Do Reasoning-Enabled Large Language Models Improve Drug–Target Interaction Prediction? A Comparative Study"
<img width="1018" height="384" alt="Figure_Pipeline" src="https://github.com/user-attachments/assets/5b7eb1c6-85ae-499f-84d0-8b8865e15897" />


Instructions to run the code:

Direct:

python run_experiments.py  
--train_emb /train_with_emb.parquet 
--test_emb /test_with_emb.parquet 
--base_model /Qwen 
--output_root /Results_Qwen/direct 
--prompt_style direct 
--feature_mode all   
--seeds 42 123 999

Chain-of-Thought:

python run_experiments.py  
--train_emb /train_with_emb.parquet 
--test_emb /test_with_emb.parquet 
--base_model /Qwen 
--output_root /Results_Qwen/cot 
--prompt_style cot 
--feature_mode all  
--seeds 42 123 999

Synthetic Rationale:

python run_experiments_SR.py  
--train_emb /train_with_emb.parquet 
--test_emb /test_with_emb.parquet 
--base_model /Qwen
--output_root /Results_Qwen/SR 
--prompt_style synthetic_rationale 
--feature_mode all   
--seeds 42 123 999


Note: 
1) Type of prompt_style: direct, synthetic_rationale, cot
2) feature_mode: all, ppi_only, seq_only, no_emb, no_ppi, no_seq








