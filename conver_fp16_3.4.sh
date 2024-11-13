model_deploy.py \
    --mlir new51200_sim1_0321.mlir  \
    --quantize F16 \
    --chip bm1688 \
    --model yana_qat_51200_bm1688_f16_sym.bmodel \
    --num_core 1 \
    --opt 2 \
    --quant_input \
    --quant_output \
    --customization_format BGGR_RAW \
    --quant_input_list 2,3 \
    --fuse_preprocess \
    --addr_mode io_tag \
    # --compress_mode all \
  
    # --test_input  new51200_sim1_0321_in_f32.npz \
