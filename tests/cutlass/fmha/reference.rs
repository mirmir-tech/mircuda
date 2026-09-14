use mircuda::bf16;

pub(super) struct ReferenceShape {
    pub query_lengths: [usize; 2],
    pub context_lengths: [usize; 2],
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
}

pub(super) fn attention(
    query: &[bf16],
    keys: &[bf16],
    values: &[bf16],
    shape: ReferenceShape,
) -> Vec<bf16> {
    let mut output = Vec::with_capacity(query.len());
    let mut query_offset = 0;
    let mut context_offset = 0;
    for (queries, context) in shape.query_lengths.into_iter().zip(shape.context_lengths) {
        for row in 0..queries {
            let visible = context - queries + row + 1;
            for head in 0..shape.query_heads {
                let kv_head = head / (shape.query_heads / shape.kv_heads);
                let query_start =
                    ((query_offset + row) * shape.query_heads + head) * shape.head_dim;
                let scores = (0..visible)
                    .map(|token| {
                        let key_start =
                            ((context_offset + token) * shape.kv_heads + kv_head) * shape.head_dim;
                        (0..shape.head_dim)
                            .map(|item| {
                                f64::from(query[query_start + item].to_f32())
                                    * f64::from(keys[key_start + item].to_f32())
                            })
                            .sum::<f64>()
                            * f64::from(shape.scale)
                    })
                    .collect::<Vec<_>>();
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights =
                    scores.iter().map(|value| (value - maximum).exp()).collect::<Vec<_>>();
                let denominator = weights.iter().sum::<f64>();
                for item in 0..shape.head_dim {
                    let value = weights
                        .iter()
                        .enumerate()
                        .map(|(token, weight)| {
                            let index = ((context_offset + token) * shape.kv_heads + kv_head)
                                * shape.head_dim
                                + item;
                            weight / denominator * f64::from(values[index].to_f32())
                        })
                        .sum::<f64>();
                    #[allow(clippy::cast_possible_truncation)]
                    output.push(bf16::from_f32(value as f32));
                }
            }
        }
        query_offset += queries;
        context_offset += context;
    }
    output
}
