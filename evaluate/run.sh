python generate_ai_test_cases.py \
    --preference robustness \
    --base_dataset 75k \
    --api_key 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\
    --start_index 0 \
    --end_index 200

python generate_ai_test_cases.py \
    --preference functionality \
    --base_dataset 75k \
    --api_key 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\
    --start_index 0 \
    --end_index 200

python generate_ai_test_cases.py \
    --preference robustness \
    --base_dataset 110k \
    --api_key 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\
    --start_index 0 \
    --end_index 200

python generate_ai_test_cases.py \
    --preference functionality \
    --base_dataset 110k \
    --api_key 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\
    --start_index 0 \
    --end_index 200
    
# python eval_result.py --preference robustness --api_key $FSQ_GPT_KEY1 --base_dataset 75k --max_index 120 --use_ai_test --start_index 0 --end_index 10 