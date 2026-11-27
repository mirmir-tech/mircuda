/// One benchmarked `weight[N,K] * input[K]` shape.
#[derive(Clone, Copy)]
pub struct Case {
    pub name: &'static str,
    pub n: usize,
    pub k: usize,
}

pub const CASES: &[Case] = &[
    Case { name: "router", n: 128, k: 2_816 },
    Case { name: "router-32-2048", n: 32, k: 2_048 },
    Case { name: "router-32-4096", n: 32, k: 4_096 },
    Case { name: "router-32-7168", n: 32, k: 7_168 },
    Case {
        name: "router-128-2048",
        n: 128,
        k: 2_048,
    },
    Case {
        name: "router-128-4096",
        n: 128,
        k: 4_096,
    },
    Case {
        name: "router-128-7168",
        n: 128,
        k: 7_168,
    },
    Case {
        name: "router-256-2048",
        n: 256,
        k: 2_048,
    },
    Case {
        name: "router-256-4096",
        n: 256,
        k: 4_096,
    },
    Case {
        name: "router-256-7168",
        n: 256,
        k: 7_168,
    },
    Case {
        name: "shared-gate-up-2048",
        n: 512,
        k: 2_048,
    },
    Case {
        name: "shared-down-2048",
        n: 2_048,
        k: 512,
    },
    Case {
        name: "mixer-qkv-2048",
        n: 8_192,
        k: 2_048,
    },
    Case {
        name: "mixer-gate-2048",
        n: 4_096,
        k: 2_048,
    },
    Case {
        name: "mixer-parameters-2048",
        n: 32,
        k: 2_048,
    },
    Case {
        name: "mixer-output-2048",
        n: 2_048,
        k: 4_096,
    },
    Case { name: "gpt-oss-qkv", n: 5_120, k: 2_880 },
    Case {
        name: "gpt-oss-attention-output",
        n: 2_880,
        k: 4_096,
    },
    Case { name: "gpt-oss-router", n: 32, k: 2_880 },
    Case {
        name: "gpt-oss-output-head",
        n: 201_088,
        k: 2_880,
    },
    Case {
        name: "attention-qkv-2560",
        n: 6_144,
        k: 2_560,
    },
    Case {
        name: "attention-output-2560",
        n: 2_560,
        k: 4_096,
    },
    Case {
        name: "dense-gate-up-2560",
        n: 19_456,
        k: 2_560,
    },
    Case {
        name: "dense-down-2560",
        n: 2_560,
        k: 9_728,
    },
    Case {
        name: "output-head-2560",
        n: 151_936,
        k: 2_560,
    },
    Case { name: "dense-down", n: 2_816, k: 2_112 },
    Case {
        name: "attention-output",
        n: 2_816,
        k: 4_096,
    },
    Case {
        name: "dense-gate-up",
        n: 4_224,
        k: 2_816,
    },
    Case { name: "local-qkv", n: 8_192, k: 2_816 },
    Case { name: "global-qkv", n: 10_240, k: 2_816 },
    Case {
        name: "output-head",
        n: 262_144,
        k: 2_816,
    },
];
