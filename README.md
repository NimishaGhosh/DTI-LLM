# DTI-LLM
LLM based reasoning model to predict Drug-Target Interaction

Instructions to run the code:
Direct:
python run_experiments.py  \
--train_emb /train_with_emb.parquet \
--test_emb /test_with_emb.parquet \
--base_model /Qwen \
--output_root /Results_Qwen/direct \
--prompt_style direct \
--feature_mode all \  
--seeds 42 123 999

Chain-of-Thought:
python run_experiments.py  \
--train_emb /train_with_emb.parquet \
--test_emb /test_with_emb.parquet \
--base_model /Qwen \
--output_root /Results_Qwen/cot \
--prompt_style cot \ 
--feature_mode all \  
--seeds 42 123 999

Synthetic Rationale:
python run_experiments_SR.py  \
--train_emb /train_with_emb.parquet \
--test_emb /test_with_emb.parquet \
--base_model /Qwen \
--output_root /Results_Qwen/SR \
--prompt_style synthetic_rationale \
--feature_mode all \  
--seeds 42 123 999


Note: 
1) Type of prompt_style: direct, synthetic_rationale, cot
2) feature_mode: all, ppi_only, seq_only, no_emb, no_ppi, no_seq








