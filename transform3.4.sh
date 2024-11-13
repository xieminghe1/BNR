model_transform.py \
    --model_name new51200_sim1_0321 \
    --model_def ../reshape_test_down.onnx \
    --keep_aspect_ratio \
    --input_shapes [[1,4,720,1280],[1,16,360,640],[1,1,360,640],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,4,8,8],[1,1,1,1],[1,1,1,1]]  \
    --pixel_format bggr \
    --preprocess_list 1 \
    --add_postprocess bnr \
    --mlir new51200_sim1_0321.mlir
    # --test_input /workspace/yana_fp16_51200/cali/input_1.npz \
    # --test_result new51200_sim1_0321_top_output.npz 
