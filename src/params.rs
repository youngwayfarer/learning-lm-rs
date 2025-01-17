use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // TODO:("实现从safetensors文件的模型参数加载");
        let get_tensor = |name: &str| {
            let tensor = safetensor.tensor(name).unwrap();
            let data_u8 = tensor.data();
            // convert u8 to f32
            let data_f32: Vec<f32> = data_u8
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Tensor::new(data_f32, &tensor.shape().to_vec())
        };

        LLamaParams {
            embedding_table: get_tensor("lm_head.weight"),
            rms_att_w: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.input_layernorm.weight")))
                .collect(),
            
            wq: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.q_proj.weight")))
                .collect(),
            wk: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.k_proj.weight")))
                .collect(),
            wv: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.v_proj.weight")))
                .collect(),
            wo: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.o_proj.weight")))
                .collect(),
            rms_ffn_w: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.post_attention_layernorm.weight")))
                .collect(),
            w_up: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.mlp.up_proj.weight")))
                .collect(),
            w_down: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.mlp.down_proj.weight")))
                .collect(),
            w_gate: (0..config.num_hidden_layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.mlp.gate_proj.weight")))
                .collect(),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }

    }
}
