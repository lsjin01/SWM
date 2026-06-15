"""lm_head lora_B rows 31744:31750 의 norm 확인"""
import struct, json, math

path = '/scratch/mip25/sjLee/SWM/ckpts/checkpoint_files/SFT_models/square/lora_adapter/adapter_model.safetensors'
with open(path, 'rb') as f:
    header_size = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(header_size))
    data_start = 8 + header_size
    
    def read_bf16_tensor(key):
        meta = header[key]
        shape = meta['shape']
        begin, end = meta['data_offsets']
        f.seek(data_start + begin)
        raw = f.read(end - begin)
        arr = struct.unpack(f'<{len(raw)//2}H', raw)
        floats = []
        for h in arr:
            b = h << 16
            floats.append(struct.unpack('f', struct.pack('I', b))[0])
        return floats, shape
    
    # lm_head lora_B shape: [32064, 32]
    lora_B, (rows, rank) = read_bf16_tensor('base_model.model.language_model.lm_head.lora_B.weight')
    
    # rows 31744:31750 의 norm
    print(f"lora_B shape: {rows}x{rank}")
    print(f"Rows 31744:31750 norms:")
    for r in range(31744, min(31750, rows)):
        row = lora_B[r*rank:(r+1)*rank]
        norm = math.sqrt(sum(x*x for x in row))
        print(f"  row {r}: norm={norm:.6f}, values={[round(x,4) for x in row[:5]]}...")
    
    # 전체 행 norm 분포 (샘플)
    all_row_norms = [math.sqrt(sum(lora_B[r*rank:(r+1)*rank][i]**2 for i in range(rank))) for r in range(0, rows, 1000)]
    print(f"\nRow norm distribution (every 1000th row): min={min(all_row_norms):.4f} max={max(all_row_norms):.4f}")
    
    # 전체 lm_head lora_A x lora_B product norm (approximate)
    lora_A, (rank2, cols) = read_bf16_tensor('base_model.model.language_model.lm_head.lora_A.weight')
    print(f"\nlora_A shape: {rank2}x{cols}")
    
    # 첫 10행 lora_B의 norm 체크
    alpha, r = 16, 32
    scaling = alpha / r  # 0.5
    print(f"\nScaling factor: {scaling}")
    
    # lora_B[0:5] × lora_A norm estimate
    # row 0 of B: shape [rank]
    # lora_A: [rank, cols] → row r of B × lora_A = vector of size cols
    row0 = lora_B[:rank]  # first row of B
    delta_row0 = [scaling * sum(row0[k] * lora_A[k*cols + j] for k in range(rank)) for j in range(min(5, cols))]
    print(f"Estimated delta for lm_head row 0 (first 5 cols): {[round(x,6) for x in delta_row0]}")
    
    row_31744 = lora_B[31744*rank:31745*rank]
    delta_row31744 = [scaling * sum(row_31744[k] * lora_A[k*cols + j] for k in range(rank)) for j in range(min(5, cols))]
    print(f"Estimated delta for lm_head row 31744 (first 5 cols): {[round(x,6) for x in delta_row31744]}")

